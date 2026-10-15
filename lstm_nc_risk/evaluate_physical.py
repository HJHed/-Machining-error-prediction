"""Evaluate frozen physical-risk flags and a freshly trained NC error model."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment import KEYS, ROOT, alert_metrics, distance_metrics, identities


PREVIOUS=ROOT/'lstm_nc_risk/reference_previous_nc_test_predictions.csv'


def run(out):
    previous=pd.read_csv(PREVIOUS)
    fresh=pd.read_csv(out/'test_weighted_predictions.csv')
    risk=pd.read_csv(out/'test_point_weights.csv')
    assert identities(previous)==identities(fresh)==identities(risk)
    assert len(fresh)==1279
    data=fresh.merge(previous[KEYS+['distance_mm']+[f'true_{axis}' for axis in 'xyz']],
                     on=KEYS,validate='one_to_one',suffixes=('_new','_previous'))
    data=data.merge(risk[KEYS+['high_risk','physical_risk_score']],on=KEYS,validate='one_to_one')
    for axis in 'xyz':
        np.testing.assert_allclose(data[f'true_{axis}_new'],data[f'true_{axis}_previous'],atol=2e-7,rtol=0)
    old_fail=data.distance_mm_previous.to_numpy(float)>=.04
    new_fail=data.distance_mm_new.to_numpy(float)>=.04
    high=data.high_risk.to_numpy(bool)
    train_risk=pd.read_csv(out/'train_point_weights.csv')
    val_risk=pd.read_csv(out/'val_point_weights.csv')
    raw_risk=pd.read_csv(out/'all_point_preprocessing_weights.csv')
    history=pd.read_csv(out/'training_history.csv')
    summary={
        'risk_definition':'extreme measured acceleration/pose jump, or coincident high motion and NC corner/next-segment change; train-frame 75/90/95/99th percentile thresholds per NC program',
        'risk_score_kind':'physical-dynamics severity, not model-failure probability',
        'risk_threshold_selection':'fixed physical rule from training inputs only; no target recall optimization',
        'data_split':'same 17 JSON, phase_blocks, seed 2031, 10-frame sequence',
        'previous_nc_model':distance_metrics(data.distance_mm_previous),
        'fresh_model':distance_metrics(data.distance_mm_new),
        'train_high_risk':int(train_risk.high_risk.sum()),'train_points':len(train_risk),
        'validation_high_risk':int(val_risk.high_risk.sum()),'validation_points':len(val_risk),
        'test_high_risk':int(high.sum()),'test_points':len(high),
        'raw_high_risk':int(raw_risk.high_risk.sum()),'raw_points':len(raw_risk),
        'risk_against_previous_failures':alert_metrics(old_fail,high),
        'risk_against_fresh_failures':alert_metrics(new_fail,high),
        'previous_failed_now_correct':int((old_fail&~new_fail).sum()),
        'previous_correct_now_failed':int((~old_fail&new_fail).sum()),
        'best_validation_epoch':int(history.val_position_distance_ratio.argmax()+1),
        'best_validation_accuracy_pct':float(history.val_position_distance_ratio.max()*100),
        'test_reuse_note':'This same test set was viewed during prior project iterations; the comparison is exploratory.',
    }
    assert summary['fresh_model']['pass']==summary['previous_nc_model']['pass']+summary['previous_failed_now_correct']-summary['previous_correct_now_failed']
    data['previous_failed']=old_fail;data['fresh_failed']=new_fail
    data.to_csv(out/'physical_risk_point_comparison.csv',index=False,encoding='utf-8-sig')
    (out/'evaluation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    old=summary['previous_nc_model'];new=summary['fresh_model']
    prior=summary['risk_against_previous_failures'];current=summary['risk_against_fresh_failures']
    report=f'''# NC 融合 + 物理运动风险：重新训练结果

本次以先前 **81.16%** 的 NC 融合版 `main.py` 与 `motion_risk.py` 为起点。未使用上一轮基于模型失败概率的重训模块，也没有按照失败召回率下调警报阈值。数据为同一 17 个 JSON、相同 `phase_blocks` 划分、种子 2031 和 10 帧序列；训练/验证/测试可预测点数为 6,863 / 1,279 / 1,279。达标定义：三维位置预测残差 < 0.04 mm。

“高风险”现在专指**异常刀具运动或 NC 路径急变**：每个 NC 程序只利用训练帧标定加速度、刀位跳变、NC 转角和下一段长度的 75/90/95/99 分位数。极端加速度或跳变可单独触发，其余情况要求运动与路径变化同时偏高。转速、转速加速度、下一步进给变化和旋转变化在这批数据几乎全为零，没有作为主要风险证据。连续物理风险分数进入静态输入，严格规则决定高风险标签；高风险点损失权重仅温和提高。此分数不是“误差模型失败概率”。阈值在查看验证/测试误差之前已固定。

| 指标 | 结果 |
|---|---:|
| 上一 NC 模型测试达标 | {old['pass']}/{old['total']}，{old['accuracy_pct']:.2f}% |
| **本次重训模型测试达标** | **{new['pass']}/{new['total']}，{new['accuracy_pct']:.2f}%** |
| 达标率变化 | {new['accuracy_pct']-old['accuracy_pct']:+.2f} 个百分点 |
| 训练集高风险 | {summary['train_high_risk']}/{summary['train_points']}，{100*summary['train_high_risk']/summary['train_points']:.2f}% |
| 验证集高风险 | {summary['validation_high_risk']}/{summary['validation_points']}，{100*summary['validation_high_risk']/summary['validation_points']:.2f}% |
| **测试集高风险** | **{summary['test_high_risk']}/{summary['test_points']}，{100*summary['test_high_risk']/summary['test_points']:.2f}%** |
| 上一模型失败点被标高风险 | {prior['failure_flagged']}/{prior['failures']}，{prior['failure_recall_pct']:.2f}% |
| 上一模型正确点被标高风险 | {prior['correct_flagged']}/{prior['correct']}，{prior['correct_false_alert_pct']:.2f}% |
| 本次模型失败点被标高风险 | {current['failure_flagged']}/{current['failures']}，{current['failure_recall_pct']:.2f}% |
| 本次模型正确点被标高风险 | {current['correct_flagged']}/{current['correct']}，{current['correct_false_alert_pct']:.2f}% |
| 上一模型失败 → 本次正确 | {summary['previous_failed_now_correct']} |
| 上一模型正确 → 本次失败 | {summary['previous_correct_now_failed']} |
| 最高验证达标率 | {summary['best_validation_accuracy_pct']:.2f}%（第 {summary['best_validation_epoch']} 轮） |

失败预测与加工物理风险并非同义。严格物理风险规则预期不会覆盖大多数预测失败点；低覆盖不能仅凭预测失败标签判断规则错误。相反，高风险规则的物理有效性仍需机床动态、轮廓误差或工艺事件等独立证据验证。同一测试集此前已被反复查看，本次精度对照属于探索性复用。
'''
    (out/'report.md').write_text(report,encoding='utf-8')
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    labels=['训练','验证','测试'];counts=[len(train_risk),len(val_risk),len(risk)]
    highs=[int(train_risk.high_risk.sum()),int(val_risk.high_risk.sum()),int(high.sum())]
    axes[0].bar(labels,[100*h/n for h,n in zip(highs,counts)],color='#2e8f78')
    axes[0].set(ylabel='高风险点占比 / %',title='固定物理风险规则')
    axes[1].bar(['上一 NC 模型','物理风险重训'],[old['accuracy_pct'],new['accuracy_pct']],color=['#7297bb','#2e8f78'])
    axes[1].set(ylim=(0,100),ylabel='测试达标率 / %',title='三维残差 < 0.04 mm')
    matrix=np.array([[current['correct_unflagged'],current['correct_flagged']],
                     [current['failure_missed'],current['failure_flagged']]])
    axes[2].imshow(matrix,cmap='Greens')
    for i in range(2):
        for j in range(2):axes[2].text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=14)
    axes[2].set_xticks([0,1],['未标','高风险']);axes[2].set_yticks([0,1],['预测正确','预测失败'])
    axes[2].set_title('物理风险与本次预测结果')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--result-dir',type=Path,required=True)
    run(parser.parse_args().result_dir)
