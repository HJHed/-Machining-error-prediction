"""Audit identical-data Exp1 comparisons; test labels never select parameters."""
import argparse
import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

KEYS = ['source_file_name', 'mapped_file_index', 'mapped_nc_line',
        'mapped_frame_num', 'mapped_frame_id', 'point_index']
ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / 'results/exp1_gru_highfreq_300_retrain'
if not (BASELINE / 'tool_error_test_predictions_v10_hybrid_spatiotemporal.csv').exists():
    BASELINE = ROOT / 'results/exp1_gru_highfreq_300'
PRED_FILE = 'tool_error_test_predictions_v10_hybrid_spatiotemporal.csv'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identities(table):
    assert not table.duplicated(KEYS).any(), 'Duplicate point identity'
    return set(map(tuple, table[KEYS].to_numpy()))


def check_baseline_and_record(g, d):
    old = pd.read_csv(BASELINE / PRED_FILE)
    assert identities(old) == identities(d['test_meta']), 'Test points differ from reproduced Exp1'
    assert len(old) == 1279, 'Expected 1279 test points'
    splits = [d[s+'_meta'] for s in ['train', 'val', 'test']]
    frame_keys = [k for k in KEYS if k != 'point_index']
    frame_sets = [set(map(tuple, t[frame_keys].to_numpy())) for t in splits]
    assert all(not frame_sets[i] & frame_sets[j] for i,j in [(0,1),(0,2),(1,2)])
    assert len(set(g['TARGET_COLS']) & set(d['static_feature_cols'] + d['state_feature_cols'])) == 0
    files = [{'name': Path(p).name, 'sha256': sha(p)} for p in d['json_paths']]
    audit = {
        'base_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'baseline_csv_sha256': sha(BASELINE / PRED_FILE),
        'baseline_accuracy_pct': float((old.actual_position_distance < .04).mean()*100),
        'dataset_dir': str(Path(g['JSON_DIR']).resolve()), 'data_files': files,
        'raw_points': len(d['point_df']), 'split_mode': g['SPLIT_MODE'],
        'seed': g['RANDOM_SEED'], 'time_steps': g['TIME_STEPS'],
        'sequence_group_columns': d['frame_group_cols'],
        'split_sample_counts': {s:len(d[s+'_meta']) for s in ['train','val','test']},
        'batch_size': g['BATCH_SIZE'], 'epochs': g['EPOCHS'], 'patience': g['PATIENCE'],
        'static_feature_columns': d['static_feature_cols'], 'sequence_feature_columns': d['state_feature_cols'],
        'no_test_selection': True,
    }
    (Path(g['OUTPUT_DIR'])/'experiment_manifest.json').write_text(
        json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Verified exactly the same 1279 test point identities as 81.16% Exp1.',flush=True)


def distance_metrics(values):
    values = np.asarray(values,float)
    return {'total': len(values), 'pass': int((values < .04).sum()),
            'fail': int((values >= .04).sum()), 'accuracy_pct': float((values < .04).mean()*100),
            'mean_mm': float(values.mean()), 'p95_mm': float(np.quantile(values,.95))}


def alert_metrics(fail, high):
    fail = np.asarray(fail,bool); high = np.asarray(high,bool)
    tp = int((fail & high).sum()); fp = int((~fail & high).sum())
    return {'failures': int(fail.sum()), 'correct': int((~fail).sum()),
            'failure_flagged':tp, 'failure_missed': int((fail & ~high).sum()),
            'correct_flagged':fp, 'correct_unflagged':int((~fail & ~high).sum()),
            'failure_recall_pct':float(tp / max(1,fail.sum())*100),
            'correct_false_alert_pct':float(fp / max(1,(~fail).sum())*100),
            'alert_precision_pct':float(tp / max(1,high.sum())*100)}


def architecture(out):
    fig,ax = plt.subplots(figsize=(15,7),facecolor='#f7f9fc')
    ax.set(xlim=(0,15),ylim=(0,7)); ax.axis('off')
    ax.text(.3,6.65,'81.16% Exp1 基线 + NC 融合 + 风险预处理',fontsize=20,weight='bold',color='#16384d')
    def box(x,y,w,h,title,body,new=False):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',
            facecolor='#e8f6f2' if new else '#e9eff8',edgecolor='#159a7e' if new else '#7191b8'))
        ax.text(x+w/2,y+h-.28,title,ha='center',fontsize=11,weight='bold')
        ax.text(x+w/2,y+h/2-.15,body,ha='center',va='center',fontsize=10,linespacing=1.6)
    def arrow(a,b):
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#61748b','lw':1.6})
    box(.3,3.55,2.7,1.6,'相同 17 个 JSON + NC','保持 phase_blocks 划分\n1,279 个相同测试点\nNC 按源文件索引/行号对齐')
    box(3.55,4.6,3.0,1.5,'新增：训练前风险学习','训练集 5 折阶段块留出\n教师残差 ≥ 0.04 mm 为难点\n相邻 10 帧排除',True)
    box(3.55,2.1,3.0,1.65,'新增：NC + 运动风险','29 个 NC 指令/刀路特征\n运动速度、加速度、姿态、温度\n输出风险评分 p 与逐点权重 w',True)
    box(7.1,4.6,3.0,1.5,'静态分支 + 新通道','原 Exp1 静态网络\n+ NC 特征 + p/w/运动特征',True)
    box(7.1,2.1,3.0,1.65,'GRU 注意力 + NC 时序','原 GRU 96 → 48\n10 帧状态 + 29 个 NC 特征\n保持原注意力结构',True)
    box(10.7,3.55,3.6,1.6,'融合及三轴误差输出','保持原 Exp1 输出网络\n损失乘训练点权重 w\n预测点位置 = 帧姿态 + 误差')
    box(10.7,.5,3.6,1.5,'验证后冻结模型与阈值','校准及警报阈值只用验证集\n最终只评估相同测试点',True)
    arrow((3,4.8),(3.55,5.2)); arrow((5.05,4.6),(5.05,3.75)); arrow((3,3.9),(3.55,2.9))
    arrow((6.55,3.2),(7.1,5.15)); arrow((6.55,2.6),(7.1,2.9))
    arrow((10.1,5.2),(10.7,4.8)); arrow((10.1,2.9),(10.7,3.9)); arrow((12.5,3.55),(12.5,2))
    ax.text(.3,.7,'绿色为新增/扩展部分。风险分类器不接收验证集、测试集真实误差。\n警报标签与误差预测是两个输出，报告分别计算召回、误报和准确率变化。',fontsize=10,color='#536376')
    fig.savefig(out/'architecture.png',dpi=170,bbox_inches='tight'); plt.close(fig)


def report(out):
    old = pd.read_csv(BASELINE/PRED_FILE)
    new = pd.read_csv(out/PRED_FILE)
    weights = pd.read_csv(out/'test_point_weights.csv')
    assert identities(old) == identities(new) == identities(weights)
    oldcols = KEYS + ['actual_position_distance'] + [f'true_{a}' for a in 'xyz']
    baseline = old[oldcols].rename(columns={'actual_position_distance':'baseline_distance_mm',
        **{f'true_{a}':f'baseline_true_{a}' for a in 'xyz'}})
    data = new.merge(baseline,on=KEYS,validate='one_to_one').merge(
        weights[KEYS+['high_risk']],on=KEYS,validate='one_to_one')
    for a in 'xyz':
        np.testing.assert_allclose(data[f'true_{a}'],data[f'baseline_true_{a}'],atol=2e-7,rtol=0)
    oldfail = data.baseline_distance_mm >= .04; newfail = data.actual_position_distance >= .04
    high = data.high_risk.astype(bool)
    data['baseline_failed'] = oldfail; data['new_failed'] = newfail
    data['rescued'] = oldfail & ~newfail; data['new_failure'] = ~oldfail & newfail
    all_weights = pd.read_csv(out/'all_point_preprocessing_weights.csv')
    groups = []
    for source,group in data.groupby('source_file_name',sort=False):
        groups.append({'source':source,'total':len(group),
            'baseline_accuracy_pct':float((group.baseline_distance_mm<.04).mean()*100),
            'new_accuracy_pct':float((group.actual_position_distance<.04).mean()*100),
            'high_risk':int(group.high_risk.sum()),'rescued':int(group.rescued.sum()),
            'new_failure':int(group.new_failure.sum())})
    history = pd.read_csv(out/'training_history.csv')
    summary = {'baseline':distance_metrics(data.baseline_distance_mm),
        'new':distance_metrics(data.actual_position_distance),
        'high_risk_test':int(high.sum()), 'high_risk_test_pct':float(high.mean()*100),
        'risk_against_baseline':alert_metrics(oldfail,high),
        'risk_against_new_model':alert_metrics(newfail,high),
        'rescued_points':int((oldfail & ~newfail).sum()),
        'new_failure_points':int((~oldfail & newfail).sum()),
        'still_failed_points':int((oldfail & newfail).sum()),
        'still_correct_points':int((~oldfail & ~newfail).sum()),
        'rescued_baseline_failures_pct':float((oldfail & ~newfail).sum()/oldfail.sum()*100),
        'new_failure_baseline_correct_pct':float((~oldfail & newfail).sum()/(~oldfail).sum()*100),
        'raw_points':len(all_weights), 'high_risk_all_raw':int(all_weights.high_risk.sum()),
        'best_validation_epoch':int(history.val_position_distance_ratio.argmax()+1),
        'best_validation_accuracy_pct':float(history.val_position_distance_ratio.max()*100),
        'per_source':groups,
        'threshold':json.loads((out/'risk_alert_threshold.json').read_text(encoding='utf-8'))}
    assert summary['new']['pass'] == summary['baseline']['pass'] + summary['rescued_points'] - summary['new_failure_points']
    data.to_csv(out/'baseline_vs_nc_risk_points.csv',index=False,encoding='utf-8-sig')
    (out/'evaluation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    b,n = summary['baseline'],summary['new']; r=summary['risk_against_baseline']
    text=f'''# Exp1 + NC 融合及风险预处理结果

数据为仓库 data/hybrid_json 的 17 个 JSON（9,908 原始点），使用相同 phase_blocks、10 帧时序、种子 2031。基线为这次从头复现的 Exp1 81.16%。测试点标识及真实误差逐点一致。

| 指标 | 结果 |
|---|---|
| 原模型测试达标率 | {b['pass']}/{b['total']}，{b['accuracy_pct']:.4f}% |
| NC + 风险模型测试达标率 | {n['pass']}/{n['total']}，{n['accuracy_pct']:.4f}% |
| 测试点标记高风险 | {summary['high_risk_test']}/{n['total']}，{summary['high_risk_test_pct']:.2f}% |
| 原模型失败点被标记高风险 | {r['failure_flagged']}/{r['failures']}，{r['failure_recall_pct']:.2f}% |
| 原模型正确点被误标高风险 | {r['correct_flagged']}/{r['correct']}，{r['correct_false_alert_pct']:.2f}% |
| 原模型失败 → 新模型正确 | {summary['rescued_points']}/{b['fail']}，{summary['rescued_baseline_failures_pct']:.2f}% |
| 原模型正确 → 新模型失败 | {summary['new_failure_points']}/{b['pass']}，{summary['new_failure_baseline_correct_pct']:.2f}% |
| 最高验证达标率 | {summary['best_validation_accuracy_pct']:.2f}%（第 {summary['best_validation_epoch']} 轮） |

达标定义：预测点位置与真实点位置的三维距离小于 0.04 mm。风险标记与误差预测是否达标分开统计。

风险教师只使用训练集，在每个 NC 程序内划连续阶段块做 5 折留出，教师训练/内部验证与留出集之间排除相邻 10 帧。教师采用 Exp1 的 GRU 注意力架构（含 NC，尚未加入风险），100 轮上限。训练权重使用风险分类器折外评分；验证/测试由完整训练集拟合的分类器推理。权重为 (1 + 1.5p + 0.5shock) / 训练权重均值，既参与损失加权，也作为静态网络输入。

警报阈值仅按新模型验证失败点选择，目标召回至少 85%；测试真实误差未用于教师标签、风险训练、模型选择、校准或阈值选择。高召回可能同时标记很多正确点，见表中误报率。

正式网络保持 Exp1 的 300 轮上限、batch 128、patience 60 及其他原参数，仅扩展输入并加入风险权重。本评估是同工件内阶段留出，数据与基线完全一致。

详细来源和输入哈希：experiment_manifest.json、nc_alignment.json；逐点结果：baseline_vs_nc_risk_points.csv；模型：nc_risk_model.keras；预处理、风险模型和冻结校准：motion_weighted_preprocessing.pkl。
'''
    (out/'report.md').write_text(text,encoding='utf-8')
    fig,axes = plt.subplots(2,2,figsize=(13,9))
    for values,label in [(data.baseline_distance_mm,'Exp1 基线'),(data.actual_position_distance,'NC + 风险')]:
        values=np.sort(values); axes[0,0].plot(values,np.arange(1,len(values)+1)/len(values)*100,label=label)
    axes[0,0].axvline(.04,color='#d65858',ls='--'); axes[0,0].set(xlim=(0,.15),xlabel='三维残差 / mm',ylabel='累计点占比 / %'); axes[0,0].legend()
    axes[0,1].plot(data.baseline_distance_mm,label='Exp1 基线',lw=.8,alpha=.6)
    axes[0,1].plot(data.actual_position_distance,label='NC + 风险',lw=.8,alpha=.7)
    axes[0,1].axhline(.04,color='#d65858',ls='--'); axes[0,1].set(xlabel='相同测试点序号',ylabel='三维残差 / mm'); axes[0,1].legend()
    matrix=np.array([[r['correct_unflagged'],r['correct_flagged']],[r['failure_missed'],r['failure_flagged']]])
    axes[1,0].imshow(matrix,cmap='Blues')
    for i in range(2):
        for j in range(2): axes[1,0].text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=16)
    axes[1,0].set_xticks([0,1],['未标风险','高风险']); axes[1,0].set_yticks([0,1],['原预测正确','原预测失败']); axes[1,0].set_title('风险识别与误报')
    axes[1,1].plot(history.val_position_distance_ratio*100,color='#159a7e')
    axes[1,1].set(xlabel='训练轮次（从 0 开始）',ylabel='验证集达标率 / %'); axes[1,1].set_title('训练过程，仅验证集')
    fig.tight_layout(); fig.savefig(out/'comparison.png',dpi=170); plt.close(fig)
    architecture(out)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['per_source','threshold']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--result-dir',type=Path,required=True)
    report(parser.parse_args().result_dir)
