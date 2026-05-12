import { Alert, Button, Card, Col, Row, Space, Spin, Table, Typography } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { EyeOutlined } from '@ant-design/icons';
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import DashboardTrendChart from '../components/DashboardTrendChart';
import MetricCard, { StatusTag } from '../components/MetricCard';
import StageTimeline from '../components/StageTimeline';
import { getDashboardSummary, getMetricTrends, getRecentTasks } from '../services/dashboard';
import type { DashboardSummary, MetricTrendRecord, RecentTaskRecord } from '../types';

const formatMetric = (value: number) => value.toFixed(4);

export default function DashboardPage() {
  const navigate = useNavigate();
  const [summary, setSummary] = useState<DashboardSummary>();
  const [tasks, setTasks] = useState<RecentTaskRecord[]>([]);
  const [trends, setTrends] = useState<MetricTrendRecord[]>([]);

  useEffect(() => {
    Promise.all([getDashboardSummary(), getRecentTasks(), getMetricTrends()]).then(
      ([summaryData, taskData, trendData]) => {
        setSummary(summaryData);
        setTasks(taskData);
        setTrends(trendData);
      },
    );
  }, []);

  const columns: ColumnsType<RecentTaskRecord> = [
    { title: 'iteration_id', dataIndex: 'iteration_id', width: 170 },
    { title: 'train_plan', dataIndex: 'train_plan', width: 150 },
    {
      title: 'status',
      dataIndex: 'status',
      width: 110,
      render: (value: RecentTaskRecord['status']) => <StatusTag status={value} />,
    },
    { title: 'val_ap', dataIndex: 'val_ap', render: formatMetric },
    { title: 'recall_p98', dataIndex: 'recall_p98', render: formatMetric },
    { title: 'created_at', dataIndex: 'created_at', width: 180 },
    {
      title: 'action',
      key: 'action',
      width: 120,
      render: (_, record) => (
        <Button size="small" icon={<EyeOutlined />} onClick={() => navigate(`/tasks/${record.iteration_id}`)}>
          查看任务
        </Button>
      ),
    },
  ];

  if (!summary) {
    return (
      <div className="center-box">
        <Spin tip="加载 Dashboard" />
      </div>
    );
  }

  return (
    <Space direction="vertical" size={18} className="page-stack">
      <div className="page-intro">
        <div>
          <Typography.Title level={2}>Dashboard 首页</Typography.Title>
          <Typography.Paragraph type="secondary">
            查看当前生产模型、最近迭代任务、关键指标趋势和流程状态。
          </Typography.Paragraph>
        </div>
      </div>

      <Row gutter={[16, 16]}>
        <Col xs={24} sm={12} xl={4}>
          <MetricCard title="当前生产模型" value={summary.productionModel} />
        </Col>
        <Col xs={24} sm={12} xl={4}>
          <MetricCard title="最近迭代任务" value={summary.latestIteration} />
        </Col>
        <Col xs={24} sm={12} xl={4}>
          <MetricCard title="最近任务状态" value={summary.latestStatus} tag={<StatusTag status={summary.latestStatus} />} />
        </Col>
        <Col xs={24} sm={12} xl={4}>
          <MetricCard title="Val AP" value={summary.valAp} precision={4} />
        </Col>
        <Col xs={24} sm={12} xl={4}>
          <MetricCard title="Recall@P98" value={summary.recallP98} precision={4} />
        </Col>
        <Col xs={24} sm={12} xl={4}>
          <MetricCard title="Hard Recall@P98" value={summary.hardRecallP98} precision={4} />
        </Col>
      </Row>

      {summary.latestTestUnseen === 0 ? (
        <Alert
          type="warning"
          showIcon
          message="最近一次迭代缺少 unseen 泛化测试集，建议补充 split_hint=unseen 的新合成模型数据。"
        />
      ) : null}

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={14}>
          <Card title="最近任务列表" className="panel-card">
            <Table
              rowKey="iteration_id"
              columns={columns}
              dataSource={tasks}
              pagination={false}
              scroll={{ x: 900 }}
              size="middle"
            />
          </Card>
        </Col>
        <Col xs={24} xl={10}>
          <Card title="最近任务流程状态" className="panel-card">
            <StageTimeline current={8} />
          </Card>
        </Col>
      </Row>

      <Card title="核心指标趋势" className="panel-card">
        <DashboardTrendChart data={trends} />
      </Card>
    </Space>
  );
}
