"""Saved-only failure review; produce unapplied repair and stdlib-only checks."""
from common import *
from contextlib import contextmanager
from types import SimpleNamespace
import difflib

class ReusablePhase:
    def __init__(self,ledger,key):self.ledger=ledger;self.key=key;self.stack=[]
    def __enter__(self):
        self.stack.append(self.ledger.scope);self.ledger.scope=self.key
        return self
    def __exit__(self,typ,value,tb):
        self.ledger.scope=self.stack.pop()
        return False

def main():
    failure=read(ROOT/'completion.json')
    assert failure['failed'] and failure['ledger']['successful_updates']=={}
    # Reproduce the context-only failure without a model, tensor or GPU call.
    @contextmanager
    def old():yield
    c=old()
    with c:pass
    try:
        with c:pass
    except AttributeError as exc:old_error=str(exc)
    else:raise AssertionError('Expected one-shot context failure')
    stub=SimpleNamespace(scope='outer');p=ReusablePhase(stub,'recompute')
    for _ in range(2):
        with p:assert stub.scope=='recompute'
        assert stub.scope=='outer'
    with p:
        with p:assert stub.scope=='recompute'
        assert stub.scope=='recompute'
    assert stub.scope=='outer' and p.stack==[]
    try:
        with p:raise ValueError('test')
    except ValueError:pass
    assert stub.scope=='outer' and p.stack==[]
    write(ROOT/'repair-review.json',dict(original_context_failure_reproduced=old_error,reusable_context_sequential_nested_exception_checks_passed=True,new_model_calls=0,new_training=0,repair_applied_to_frozen_worker=False,full_Torch_checkpoint_validation_pending=True))
    original=(ROOT/'worker.py').read_text(encoding='utf8')
    old_block="""    @contextmanager
    def phase(self,key):
        old=self.scope;self.scope=key
        try:yield
        finally:self.scope=old
"""
    replacement="""    def phase(self,key):
        return ReusablePhase(self,key)
"""
    cls="""class ReusablePhase:
    def __init__(self,ledger,key):self.ledger=ledger;self.key=key;self.stack=[]
    def __enter__(self):
        self.stack.append(self.ledger.scope);self.ledger.scope=self.key
        return self
    def __exit__(self,typ,value,tb):
        self.ledger.scope=self.stack.pop()
        return False

"""
    assert original.count(old_block)==1
    proposed=original.replace(old_block,replacement).replace('class Ledger:',cls+'class Ledger:',1)
    # Save gradient scalar immediately, so failure of subsequent backward cannot
    # erase the only gradient-probe value (instrumentation, not another run).
    old_line="        if not finite or grad_to_first<=0:raise RuntimeError('No finite H4->predictedH1 gradient')"
    new_line="        write(ROOT/'preflight-cross-step-gradient.json',dict(finite=finite,norm=grad_to_first,passed=finite and grad_to_first>0))\n"+old_line
    assert proposed.count(old_line)==1;proposed=proposed.replace(old_line,new_line)
    compile(proposed,'proposed-worker.py','exec')
    patch=''.join(difflib.unified_diff(original.splitlines(True),proposed.splitlines(True),fromfile='frozen/worker.py',tofile='proposed/worker.py'))
    write_bytes(ROOT/'proposed-repair.patch',patch.encode())
    # Original config and all bound files remain exactly frozen.
    c=read(ROOT/'config.json');proof={b['path']:sha(b['path'])==b['sha256'] for b in c['bindings'] if Path(b['path']).is_relative_to(ROOT)}
    assert all(proof.values());write(ROOT/'frozen-source-preservation.json',proof)
    report='''本轮状态：工程预检失败；A/B训练未启动；科学假设未得到验证。waiting_user_review。

授权与冻结：本聊天实施一轮loss对照，并实际向独立分析聊天回报。唯一拟验证变量为
B在原H1残差MSE之外增加 4 * mean_{1序列,4时刻,3,96,96}(AR图像-GT)^2。
原生预条件、50/50真实历史clean/weak、原数据/划分/动作/物理/推理3去噪均未修改。
额外分支以量化前向不变的STE近似反传，使用可微四步生成历史；不是旧detach方案。
A/B计划从同一E2模型+Adam+scaler+RNG独立开始，每臂256尝试、4096原H1曝光。
B额外1024GT曝光与6144图像前向(含重算)，不等总计算。具体公式、全部上限见PLAN/config。

实际执行：只执行了工程预检。单个固定训练序列、四步原采样器与四步训练包裹采样器，
各步输出逐元素完全相等，max_abs均为0。实际33次图像/批去噪前向：原生12、包裹12、
第一次求导的checkpoint重算9。第一次H4对预测H1求梯度调用完成，随后有限/非零检查
没有触发异常，但其范数尚未落盘，不能提供数值或把整项梯度预检标为通过。
第二次完整辅助loss.backward在重入计账上下文时抛AttributeError，未完成。
梯度调用1完成+1失败；优化器尝试0、更新0、正式训练曝光0、正式评测预测0、物理步0。
预检计算涉及1段4张不同GT，其中H4被单独求导又纳入四帧均值；这不是训练曝光。

明确原因：我的Ledger.phase用@contextmanager产生一次性对象。当前Torch checkpoint
保存context_fn返回的recompute_context并在重算时重新进入。预检先autograd.grad(...,
retain_graph=True)，随后loss.backward()需要再次进入同一个对象。Python contextlib
首次进入时删除args/kwds/func，所以第二次进入报'_GeneratorContextManager'无args。
这是验证实现的错误，不能据此说loss有效/无效、模型容量不足、O旋转或B孔洞问题已解决。
原配置与所有冻结源文件保留，未自动重跑、加步数、换种子、换系数或改物理。

具体修复建议已写成未应用的proposed-repair.patch：把计账上下文改为可重复进入并用栈
恢复scope的普通context manager；及时保存第一次跨步梯度数值。它只改变计账与日志，
不改变前向、loss、梯度路径、样本或训练配方。纯stdlib检查复现旧错误并验证修复的
连续重入、嵌套和异常恢复，0模型调用。完整Torch梯度预检和A/B结果仍待批准重跑。
本轮冻结文件不被修补，若后续批准，应以明确新修订引用此失败证据，失败33次前向
仍累计入历史费用，不能把它删除或称第一次即通过。

记录的worker主体墙钟22.6863秒(不含前面的源码绑定校验/启动)，自身CPU user8.125秒、
system7.515625秒。模型CUDA，单进程BelowNormal；数值库、Torch intra/inter、OpenCV均1。
未采集独立CUDA事件计时，不能把墙钟称纯GPU时间或账单计费时间。失败前未落盘GPU峰值。
运行结束根目录约305KB，远低于1.2GB；共享约6.843GB，另有256MiB原子预留。报告和
回传元数据仍由同一写前保护计入，最终大小另存。没有删除旧证据或操作其他项目。

尚无O/C/B新1/4/8/16对照或科学验收结论，不能拿旧图冒充本轮结果。
本报告与真实工具回执将实际回传独立分析聊天，随后等待用户审批，不自动继续训练。
'''
    write_bytes(ROOT/'REPORT.txt',report.encode());write(ROOT/'review-storage.json',guard(full=True))
    print(json.dumps(dict(report=str(ROOT/'REPORT.txt'),repair=str(ROOT/'proposed-repair.patch'),frozen_sources_preserved=all(proof.values()),new_model_calls=0,storage=guard())))
if __name__=='__main__':main()
