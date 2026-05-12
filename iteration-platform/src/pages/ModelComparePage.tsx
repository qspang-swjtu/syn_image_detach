import { Alert, Card, Col, Row, Space, Spin, Table, Tag, Typography } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { mockApi } from '../services/mockApi';
import type { CompareRecord, GateResult } from '../types';

const formatValue = (value: number) => `${(value * 100).toFixed(2)}%`;

const resultColor = {
  pass: 'green',
  warning: 'gold',
  fail: 'red',
};

const columns: ColumnsType<CompareRecord> = [
  { title: 'metric', dataIndex: 'metric', width: 180 },
  { title: 'baseline', dataIndex: 'baseline', render: formatValue },
  { title: 'candidate', dataIndex: 'candidate', render: formatValue },
  {
    title: 'delta',
    dataIndex: 'delta',
    render: (value: number) => (
      <Typography.Text type={value >= 0 ? 'success' : 'danger'}>
        {value >= 0 ? '+' : ''}
        {formatValue(value)}
      </Typography.Text>
    ),
  },
  {
    title: 'result',
    dataIndex: 'result',
    render: (value: CompareRecord['result']) => <Tag color={resultColor[value]}>{value}</Tag>,
  },
];

export default function ModelComparePage() {
  const { taskId = 'iter_20260511_001' } = useParams();
  const [records, setRecords] = useState<CompareRecord[]>();
  const [gate, setGate] = useState<GateResult>();

  useEffect(() => {
    mockApi.getComparison().then((data) => {
      setRecords(data.records);
      setGate(data.gate);
    });
  }, [taskId]);

  if (!records || !gate) {
    return (
      <div className="center-box">
        <Spin tip="加载模型对比结果" />
      </div>
    );
  }

  return (
    <Space direction="vertical" size={18} className="page-stack">
      <div className="page-intro">
        <div>
          <Typography.Title level={2}>新旧模型对比</Typography.Title>
          <Typography.Paragraph type="secondary">
            baseline 与 candidate 的核心指标、gate 结果和保存建议。
          </Typography.Paragraph>
        </div>
      </div>

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={16}>
          <Card title="指标对比" className="panel-card">
            <Table
              rowKey="metric"
              columns={columns}
              dataSource={records}
              pagination={false}
              scroll={{ x: 760 }}
            />
          </Card>
        </Col>
        <Col xs={24} xl={8}>
          <Card title="Gate 结果" className="panel-card">
            <Space direction="vertical" size={14} className="page-stack">
              <Alert
                type={gate.status === 'pass' ? 'success' : gate.status === 'review' ? 'warning' : 'error'}
                showIcon
                message={`Gate: ${gate.status}`}
                description={gate.summary}
              />
              <div className="gate-checks">
                {gate.checks.map((check) => (
                  <div className="gate-check" key={check.name}>
                    <Tag color={check.passed ? 'green' : 'red'}>
                      {check.passed ? 'pass' : 'fail'}
                    </Tag>
                    <div>
                      <Typography.Text strong>{check.name}</Typography.Text>
                      <br />
                      <Typography.Text type="secondary">{check.detail}</Typography.Text>
                    </div>
                  </div>
                ))}
              </div>
            </Space>
          </Card>
        </Col>
      </Row>

      <Alert
        type="info"
        showIcon
        message="最终建议"
        description={gate.recommendation}
        className="final-advice"
      />
    </Space>
  );
}
