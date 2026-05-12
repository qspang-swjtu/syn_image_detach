import { Card, Statistic, Tag, Typography } from 'antd';
import type { ReactNode } from 'react';

interface MetricCardProps {
  title: string;
  value: string | number;
  suffix?: string;
  precision?: number;
  tag?: ReactNode;
  description?: string;
}

export default function MetricCard({ title, value, suffix, precision, tag, description }: MetricCardProps) {
  return (
    <Card className="metric-card">
      <div className="metric-card-head">
        <Typography.Text type="secondary">{title}</Typography.Text>
        {tag}
      </div>
      {typeof value === 'number' ? (
        <Statistic value={value} precision={precision ?? 4} suffix={suffix} />
      ) : (
        <Typography.Title level={4} className="metric-string">
          {value}
        </Typography.Title>
      )}
      {description ? <Typography.Text type="secondary">{description}</Typography.Text> : null}
    </Card>
  );
}

export function StatusTag({ status }: { status: 'success' | 'running' | 'failed' }) {
  const color = status === 'success' ? 'green' : status === 'running' ? 'blue' : 'red';
  return <Tag color={color}>{status}</Tag>;
}
