# audio-gain-automation

## 题面

给播客加自动增益包络后，淡入淡出处总有咔哒声：插值用线性但指数曲线也走线性，淡出到 -60 dB 时听感突变；端点外取 0 dB，前后静音被抬成满音量；db_to_linear 用 20**(db/10)，增益整体偏 3 dB；关键帧乱序不排序，包络在时间上回退；同一采样点两个关键帧不覆盖，取第一个；斜坡按帧数生成，采样率变化时斜坡时长变；跳变不插入中间点，出现咔哒声；结果超过 ±1 静默裁剪，失真不可见；批量共用包络列表，轨道之间串；linear_to_db(0) 抛除零异常而不是返回 -inf。automation/envelope.py 是核心，tests/test_envelope.py 15个红。要修的是：① exponential 在振幅域插值；② 端点外保持首尾关键帧增益；③ dB 到线性统一使用 10**(db/20)；④ 关键帧按采样点排序；⑤ 同点关键帧后值覆盖；⑥ ramp 按采样点区间生成；⑦ trim_outliers 按 max_step_db 插入中间点；⑧ 超过 ±1 的结果保留为过载标记、不静默裁剪；⑨ automate_batch 为每轨使用独立包络并保持顺序；⑩ linear_to_db 对 x<=0 返回 -inf。

## 运行方式

在仓库根目录运行：

```powershell
cd D:\gsb-20261119\audio-gain-automation
python -m pytest tests -q
```

初始失败环境必须恰好是：

```text
15 failed
```

输出尾部应为 `15 failed in ...s`，且没有 collection error、import error 或 syntax error。只需要 Python 3 和 pytest；全部离线，无外部依赖、无随机数、无 sleep。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 验收基线

`tests/test_envelope.py` 是验收基线，不得修改、删除或弱化其中任何测试。实现只改 `automation/envelope.py`，保持公开函数的签名、采样点/dB 契约、curve 名称和 `AutomationError` 语义；所有时间以采样点表示，默认绝对容差为 `1e-6`。
