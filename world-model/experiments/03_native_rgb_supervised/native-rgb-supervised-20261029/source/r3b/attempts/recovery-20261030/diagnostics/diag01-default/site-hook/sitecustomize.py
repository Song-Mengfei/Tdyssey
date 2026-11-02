import importlib.util
_td_spec=importlib.util.spec_from_file_location("tdyssey_loader_early",'__LOCAL_WORKSPACE__\\Tdyssey-raw-tl-portable-main\\tdyssey\\native_rgb_r3b_loader_diagnostics.py')
_td_module=importlib.util.module_from_spec(_td_spec)
_td_spec.loader.exec_module(_td_module)
_td_module.install_early_child_logging()
