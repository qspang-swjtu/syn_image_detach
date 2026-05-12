import { Card, Col, Row, Statistic } from 'antd';
import type { DatasetSplitSummary } from '../types';

const summaryItems: Array<{ key: keyof Pick<DatasetSplitSummary, 'allInput' | 'seenForSplit' | 'trainBase' | 'trainHard' | 'trainStage1' | 'trainStage2' | 'trainStage3Initial' | 'val' | 'testUnseen' | 'testAll' | 'reviewedPool'>; label: string }> = [
  { key: 'allInput', label: 'allInput' },
  { key: 'seenForSplit', label: 'seenForSplit' },
  { key: 'trainBase', label: 'trainBase' },
  { key: 'trainHard', label: 'trainHard' },
  { key: 'trainStage1', label: 'trainStage1' },
  { key: 'trainStage2', label: 'trainStage2' },
  { key: 'trainStage3Initial', label: 'trainStage3Initial' },
  { key: 'val', label: 'val' },
  { key: 'testUnseen', label: 'testUnseen' },
  { key: 'testAll', label: 'testAll' },
  { key: 'reviewedPool', label: 'reviewedPool' },
];

interface DatasetSummaryCardProps {
  summary: DatasetSplitSummary;
}

export default function DatasetSummaryCard({ summary }: DatasetSummaryCardProps) {
  return (
    <Card title="合并后数据统计" className="panel-card">
      <Row gutter={[12, 12]}>
        {summaryItems.map((item) => (
          <Col xs={12} md={8} xl={6} key={item.key}>
            <div className="summary-tile">
              <Statistic title={item.label} value={summary[item.key]} />
            </div>
          </Col>
        ))}
      </Row>
    </Card>
  );
}
