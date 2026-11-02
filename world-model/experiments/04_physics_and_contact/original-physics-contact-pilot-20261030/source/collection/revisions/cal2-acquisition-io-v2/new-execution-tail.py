def normalize_letters(letters):
    values = tuple(LETTERS if letters is None else letters)
    if not values or len(set(values)) != len(values) or any(letter not in LETTERS for letter in values):
        raise ValueError('Use a nonempty compact unique subset of TLOCXB, without separators')
    return tuple(letter for letter in LETTERS if letter in values)


def execution_io_paths(output, phase, run_label=None):
    output = Path(output).resolve()
    if run_label is not None:
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', run_label) or run_label.casefold() in {
                'con', 'prn', 'aux', 'nul', *[f'com{i}' for i in range(1, 10)], *[f'lpt{i}' for i in range(1, 10)]}:
            raise ValueError('Unsafe or reserved run label')
        directory = (output/'runs'/run_label).resolve()
        if not directory.is_relative_to(output):
            raise ValueError('Run directory escapes collection root')
        return dict(directory=directory, identity=directory/'identity.json',
                    ledger=directory/'ledger.json', events=directory/'events.jsonl',
                    progress=directory/'progress.json', completion=directory/'completion.json')
    return dict(directory=output, identity=output/(phase+'-run-identity.json'),
                ledger=output/'ledger.json', events=output/'events.jsonl',
                progress=output/'progress.json', completion=output/(phase+'-completion.json'))


def selected_slots(plan, phase, letters):
    if phase not in ('calibration-v1', 'calibration-v2', 'formal', 'supplement'):
        raise ValueError('Unknown frozen collection phase')
    allowed = normalize_letters(letters)
    result = [row for row in plan['slots'] if row['phase'] == phase and row['letter'] in allowed]
    if not result:
        raise ValueError('Selected frozen phase/letters have no scenes')
    return result


def _append_execution_event(path, event):
    with Path(path).open('a', encoding='utf-8') as journal:
        journal.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n')
        journal.flush()
        os.fsync(journal.fileno())


def collect(plan_path, output, phase, physics_gate, source_root, shared_roots, *,
            rules_path=None, letters=None, run_label=None, resource_policy=None, stop_file=None):
    """Frozen disjoint partitions, shared reservations, no failed-scene retry."""
    from .original_contact_pilot_resources import ResourceReservation
    import tdyssey.original_contact_pilot_resources as resources_module
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = execution_io_paths(output, phase, run_label)
    paths['directory'].mkdir(parents=True, exist_ok=True)
    if run_label is not None and (paths['identity'].exists() or paths['ledger'].exists()):
        raise FileExistsError('Named run identity/ledger already exists; no retry or reset')
    if run_label is not None and resource_policy is None:
        raise ValueError('Partitioned workers require an explicit frozen resource policy')
    letters = normalize_letters(letters)
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if identity(plan['rules']) != plan['rules_sha256'] or plan['rules'] != RULES:
        raise ValueError('Frozen scene plan/rules changed')
    gate_path = Path(physics_gate).resolve()
    gate = json.loads(gate_path.read_text(encoding='utf-8'))
    if gate.get('passed') is not True or gate.get('training_authorized') is not False:
        raise RuntimeError('A separate passed execution/physics gate is required')
    mismatches = [path for path, digest in gate.get('bindings', {}).items()
                  if not Path(path).exists() or file_sha(path) != digest]
    if mismatches:
        raise RuntimeError('Gate bindings changed before execution: '+repr(mismatches))
    module = importlib.import_module('tdyssey.native_rgb_contact_safe_physics')
    protocol_sha = module.PROTOCOL_SHA256
    if gate.get('protocol_sha256') != protocol_sha:
        raise RuntimeError('Gate protocol differs from actual safe factory')
    policy_rules = deepcopy(RULES)
    if rules_path is not None:
        revision = json.loads(Path(rules_path).read_text(encoding='utf-8'))
        allowed = {'push_advance_world_px', 'entry_advance_world_px', 'departure_distances_world_px',
                   'target_distance_world_px', 'target_change_world_px'}
        if set(revision['changes']) - allowed:
            raise ValueError('Only recorded contact-action strategy fields may change')
        policy_rules.update(revision['changes'])
    if phase == 'calibration-v2' and rules_path is None:
        raise RuntimeError('The sole revised calibration requires a frozen policy revision')
    if phase == 'supplement':
        release = output/'supplement-release.json'
        if not release.exists() or not json.loads(release.read_text(encoding='utf-8'))['released']:
            raise RuntimeError('Bounded supplement release is absent')
    selected = selected_slots(plan, phase, letters)
    stop_path = None if stop_file is None else Path(stop_file).resolve()
    run_identity = dict(phase=phase, run_label=run_label, letters=list(letters),
        selected_slot_ids=[row['slot_id'] for row in selected],
        selected_slot_ids_sha256=identity([row['slot_id'] for row in selected]),
        plan_sha256=file_sha(plan_path), physics_gate_sha256=file_sha(gate_path),
        physics_protocol_sha256=protocol_sha, physics_source_sha256=file_sha(module.__file__),
        collection_source_sha256=file_sha(__file__), resource_helper_source_sha256=file_sha(resources_module.__file__),
        policy_revision_sha256=None if rules_path is None else file_sha(rules_path),
        resource_policy_sha256=None if resource_policy is None else file_sha(resource_policy),
        resource_policy_path=None if resource_policy is None else str(Path(resource_policy).resolve()),
        stop_file=None if stop_path is None else str(stop_path),
        policy_rules_sha256=identity(policy_rules), policy_execution_version=POLICY_EXECUTION_VERSION,
        io_version=COLLECTION_IO_VERSION,
        action_contract='Requested absolute target remains the model action; endpoint/velocity intervention is separate telemetry.',
        model_calls=0, models=0, optimizer_updates=0)
    if paths['identity'].exists():
        if json.loads(paths['identity'].read_text(encoding='utf-8')) != run_identity:
            raise ValueError('Existing phase identity differs; no changed-source continuation')
    owner_prefix = f'{run_label or phase}:pid{os.getpid()}'
    def lease(owner, reserve=None):
        return ResourceReservation(output, shared_roots, resource_policy,
            owner_prefix+':'+owner, role='collection', reserve_bytes=reserve)
    def write_reserved_json(reservation, path, value):
        encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
        reservation.write_bytes(path, encoded)
    with lease('initialize', 1024**2) as reservation:
        write_reserved_json(reservation, paths['identity'], run_identity)
        ledger = json.loads(paths['ledger'].read_text(encoding='utf-8')) if paths['ledger'].exists() else dict(
            kind='original-contact-pilot-actual-cost.v2', run_label=run_label,
            attempts=[], actual_controls=0, actual_substeps=0, actual_physics_seconds=0.,
            model_calls=0, optimizer_updates=0)
        write_reserved_json(reservation, paths['ledger'], ledger)
    start = time.perf_counter()
    attempted_this_run = []
    graceful_stop = False
    failure = None
    exception_to_raise = None
    try:
        for slot in selected:
            if stop_path is not None and stop_path.exists():
                graceful_stop = True
                break
            attempt_dir = output/'trajectories'/slot['slot_id']
            claim = attempt_dir/'claim.json'
            if claim.exists():
                # Keep legacy unlabelled completed-slot references, while new
                # named workers must never claim another partition's old slot.
                if run_label is None and (attempt_dir/'metadata.json').exists():
                    continue
                raise RuntimeError('Existing scene claim cannot be retried: '+slot['slot_id'])
            with lease(slot['slot_id']) as reservation:
                attempt_dir.mkdir(parents=True, exist_ok=True)
                cost = dict(slot_id=slot['slot_id'], phase=phase, split=slot['split'],
                            letter=slot['letter'], run_label=run_label,
                            started_at_utc=utc(), reserved_controls=64, actual_controls=0,
                            status='reserved', plan_sha256=run_identity['plan_sha256'],
                            physics_protocol_sha256=protocol_sha,
                            policy_rules_sha256=identity(policy_rules),
                            policy_execution_version=POLICY_EXECUTION_VERSION)
                # Exclusive creation, even if a caller misconfigures overlapping
                # partitions. Do not overwrite any earlier attempt claim.
                with claim.open('x', encoding='utf-8') as handle:
                    json.dump(cost, handle, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                ledger['attempts'].append(cost)
                attempted_this_run.append(slot['slot_id'])
                write_reserved_json(reservation, paths['ledger'], ledger)
                states, actions, valids, images, telemetry, decisions = [], [], [], [], [], []
                env = None
                initial_identity = None
                attempt_failure = None
                failed_image_reused = False
                wall_start = time.perf_counter()
                try:
                    env = module.make_contact_safe_env(source_root, slot['letter'])
                    image = env.reset_scene(**slot['initialization'])
                    states.append(np.asarray(env.state(), np.float64).copy())
                    images.append(np.asarray(image, np.uint8).copy())
                    if not np.allclose(states[0], slot['initial_state15'], rtol=0, atol=1e-8):
                        raise RuntimeError('Initial construction differs from frozen scene')
                    initial_identity = env.initial_identity()
                    policy = PilotPolicy(env, slot, policy_rules)
                    for index in range(64):
                        before = np.asarray(env.state(), np.float64).copy()
                        action, decision = policy.next_action(index)
                        _append_execution_event(paths['events'], dict(event='control_requested', slot_id=slot['slot_id'],
                            control=index, requested_action=action.tolist(), run_label=run_label, at_utc=utc()))
                        call_start = time.perf_counter()
                        try:
                            image, info = env.step(action)
                        except Exception as exc:
                            info = getattr(exc, 'info', None) or getattr(exc, 'evidence', None) or getattr(exc, 'details', {})
                            info = dict(info or {}, valid=False, exception=repr(exc))
                            info.setdefault('substeps', getattr(env, 'last_substeps', []))
                            attempt_failure = repr(exc)
                            try:
                                if not np.isfinite(env.state()).all():
                                    raise RuntimeError('nonfinite failure state cannot render')
                                image = env.render_clean()
                            except Exception:
                                image = images[-1]
                                failed_image_reused = True
                                info['failure_RGB'] = 'Unavailable: last visible frame retained as explicitly invalid evidence, not a new observation or training frame.'
                        after = np.asarray(env.state(), np.float64).copy()
                        info = dict(info, collector_wall_seconds=time.perf_counter()-call_start)
                        duration = sum(float(row.get('dt', 0.)) for row in info.get('substeps', []) if row.get('executed'))
                        valid = bool(info.get('valid', False) and info.get('substeps')
                                     and abs(duration-.1) < 1e-7 and np.isfinite(after).all())
                        actions.append(np.asarray(action, np.float64).copy())
                        states.append(after)
                        valids.append(valid)
                        images.append(np.asarray(image, np.uint8).copy())
                        telemetry.append(_safe_json(info))
                        decisions.append(_safe_json(decision))
                        ledger['actual_controls'] += 1
                        ledger['actual_substeps'] += sum(bool(row.get('executed')) for row in info.get('substeps', []))
                        ledger['actual_physics_seconds'] += duration
                        cost['actual_controls'] = len(actions)
                        _append_execution_event(paths['events'], dict(event='control_returned', slot_id=slot['slot_id'],
                            control=index, valid=valid, actual_substeps=len(info.get('substeps', [])),
                            actual_seconds=duration, run_label=run_label, at_utc=utc()))
                        if len(actions) % 8 == 0 or not valid:
                            save_attempt(attempt_dir/'raw.npz', states, actions, valids, images, telemetry, decisions,
                                         reservation=reservation)
                        if not valid:
                            attempt_failure = attempt_failure or 'invalid_or_partial_physical_control'
                            break
                        policy.observe(before, after, info)
                except Exception as exc:
                    attempt_failure = repr(exc)
                saved = (save_attempt(attempt_dir/'raw.npz', states, actions, valids, images, telemetry, decisions,
                                      reservation=reservation) if states else None)
                complete = len(actions) == 64 and all(valids) and attempt_failure is None
                cost.update(status='completed' if complete else 'failed', finished_at_utc=utc(), actual_controls=len(actions))
                metadata = dict(slot=slot, reservation=cost, raw=saved, complete=complete,
                    failure=attempt_failure, actual_valid_controls=sum(valids), initial_identity=initial_identity,
                    wall_seconds=time.perf_counter()-wall_start, requested_controls=64,
                    policy_rules=policy_rules, policy_execution_version=POLICY_EXECUTION_VERSION,
                    protocol_sha256=protocol_sha, run_label=run_label,
                    resource_policy_sha256=run_identity['resource_policy_sha256'],
                    policy_revision_sha256=run_identity['policy_revision_sha256'],
                    model_calls=0, models=0, optimizer_updates=0,
                    contiguous_frames=not failed_image_reused,
                    invalid_failure_RGB_is_last_visible_frame=failed_image_reused,
                    hidden_replay_or_filled_frames=False, future_training_authorized=False)
                write_reserved_json(reservation, attempt_dir/'metadata.json', metadata)
                write_reserved_json(reservation, paths['ledger'], ledger)
                write_reserved_json(reservation, paths['progress'], dict(
                    phase=phase, run_label=run_label, letters=list(letters), last_slot=slot['slot_id'],
                    attempts_this_run=len(attempted_this_run), actual_controls=ledger['actual_controls'],
                    actual_substeps=ledger['actual_substeps'], updated_at_utc=utc(),
                    model_calls=0, optimizer_updates=0))
                print(json.dumps(dict(slot=slot['slot_id'], complete=complete, controls=len(actions),
                                      failure=attempt_failure, bytes=saved['bytes'] if saved else 0)), flush=True)
                if not complete:
                    raise RuntimeError('Substantive collection failure; preserve partial and stop: '+slot['slot_id']+': '+str(attempt_failure))
    except Exception as exc:
        failure = repr(exc)
        exception_to_raise = exc
    all_selected_finished = all((output/'trajectories'/row['slot_id']/'metadata.json').exists()
        and json.loads((output/'trajectories'/row['slot_id']/'metadata.json').read_text(encoding='utf-8'))['complete']
        for row in selected)
    completed = failure is None and not graceful_stop and all_selected_finished
    completion = dict(**{key: run_identity[key] for key in (
        'phase', 'run_label', 'letters', 'plan_sha256', 'physics_protocol_sha256',
        'physics_gate_sha256', 'resource_policy_sha256', 'policy_revision_sha256')},
        completed=completed, failed=failure is not None, graceful_stop=graceful_stop,
        failure=failure, finished_at_utc=utc(), wall_seconds=time.perf_counter()-start,
        selected_slots=len(selected), actual_attempts=len(attempted_this_run),
        actual_controls=ledger['actual_controls'], actual_substeps=ledger['actual_substeps'],
        unstarted_slots=[row['slot_id'] for row in selected if row['slot_id'] not in attempted_this_run],
        model_calls=0, optimizer_updates=0)
    # Small terminal evidence must survive a data-allocation failure; the data
    # guard has retained the administrative/atomic margins throughout execution.
    write_json(paths['completion'], completion)
    if exception_to_raise is not None:
        raise exception_to_raise
    return completion
