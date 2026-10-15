"""Explain which input-only kinematic/NC conditions triggered test warnings."""
import json
import pickle
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from experiment import ROOT


OUT=ROOT/'results/exp1_nc_physical_risk_20261015'


def main(out=OUT):
    artifact=pickle.loads((out/'motion_weighted_preprocessing.pkl').read_bytes())
    spec=artifact['motion_risk']['physical_risk']
    physical=pd.read_csv(out/'test_point_weights.csv')
    data=physical
    assert data[['nc_corner_degrees','nc_next_segment_length']].notna().all().all()
    ids=data.mapped_file_index.to_numpy()
    def exceed(col,q):
        cut=np.array([spec['programs'][int(i)][col]['quantiles'][q] for i in ids])
        return data[col].to_numpy(float)>cut
    a90=exceed('risk_linear_acceleration',1)
    a95=exceed('risk_linear_acceleration',2)
    a99=exceed('risk_linear_acceleration',3)
    j75=exceed('risk_pose_jump',0)
    j95=exceed('risk_pose_jump',2)
    j99=exceed('risk_pose_jump',3)
    c90=exceed('nc_corner_degrees',1)
    s95=exceed('nc_next_segment_length',2)
    triggers={
        'acceleration_above_99pct':a99,
        'pose_jump_above_99pct':j99,
        'acceleration_above_95pct_with_jump_or_nc_corner':a95&(j75|c90),
        'acceleration_above_90pct_and_jump_above_95pct':a90&j95,
        'acceleration_above_90pct_and_nc_next_segment_above_95pct':a90&s95,
    }
    union=np.logical_or.reduce(list(triggers.values()))
    np.testing.assert_array_equal(union,data.high_risk.to_numpy(bool))
    motion_only=a99|j99|(a95&j75)|(a90&j95)
    nc_necessary=union&~motion_only
    report={
        'test_points':len(data),'high_risk':int(union.sum()),
        'trigger_counts_overlapping':{k:int(v.sum()) for k,v in triggers.items()},
        'high_risk_requiring_nc_condition':int(nc_necessary.sum()),
        'explanation':'Trigger counts overlap. NC is also present in the error-network input for every point.',
    }
    (out/'physical_risk_trigger_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    summary_path=out/'evaluation_summary.json'
    summary=json.loads(summary_path.read_text(encoding='utf-8'))
    summary['physical_trigger_audit']=report
    summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    report_path=out/'report.md'
    prior=report_path.read_text(encoding='utf-8').split('\n测试点触发条件审计：')[0]
    prior+=f"\n测试点触发条件审计：{report['high_risk']} 个高风险点中，{report['high_risk_requiring_nc_condition']} 个必须依赖 NC 转角或下一段长度条件才会触发；其余由极端运动或运动组合触发。NC 特征也始终进入误差预测网络。各条件可重叠，详见 `physical_risk_trigger_audit.json`。\n"
    report_path.write_text(prior,encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--result-dir',type=Path,default=OUT)
    main(parser.parse_args().result_dir)
