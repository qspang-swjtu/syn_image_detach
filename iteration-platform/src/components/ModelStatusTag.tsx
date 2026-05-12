import { Tag } from 'antd';
import type { ModelStatus } from '../types';

const colors: Record<ModelStatus, string> = {
  production: 'green',
  candidate: 'blue',
  archived: 'default',
  failed: 'red',
};

export default function ModelStatusTag({ status }: { status: ModelStatus }) {
  return <Tag color={colors[status]}>{status}</Tag>;
}
