import { Steps } from 'antd';

const stageTitles = [
  '数据合并',
  '数据切分',
  'Stage1',
  'Stage2',
  'Replay',
  'Stage3',
  '评估',
  '新旧模型对比',
  '模型保存',
];

interface StageTimelineProps {
  current?: number;
}

export default function StageTimeline({ current = 8 }: StageTimelineProps) {
  return (
    <Steps
      size="small"
      current={current}
      items={stageTitles.map((title) => ({ title }))}
      className="stage-timeline"
    />
  );
}
