import { Table, Tag } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import type { ManifestSource } from '../types';

const columns: ColumnsType<ManifestSource> = [
  { title: 'name', dataIndex: 'name', width: 180 },
  { title: 'path', dataIndex: 'path', width: 280, ellipsis: true },
  {
    title: 'label',
    dataIndex: 'label',
    width: 90,
    render: (value: ManifestSource['label']) => (
      <Tag color={value === 1 ? 'volcano' : 'green'}>{value}</Tag>
    ),
  },
  { title: 'generator', dataIndex: 'generator', width: 120 },
  {
    title: 'split_hint',
    dataIndex: 'split_hint',
    width: 130,
    render: (value: ManifestSource['split_hint']) => <Tag color="blue">{value || 'seen'}</Tag>,
  },
  { title: 'sample_weight', dataIndex: 'sample_weight', width: 140 },
  {
    title: 'is_hard_negative',
    dataIndex: 'is_hard_negative',
    width: 150,
    render: (value: ManifestSource['is_hard_negative']) => (
      <Tag color={value === 1 ? 'orange' : 'default'}>{value ?? 0}</Tag>
    ),
  },
];

interface ManifestPreviewTableProps {
  data: ManifestSource[];
}

export default function ManifestPreviewTable({ data }: ManifestPreviewTableProps) {
  return (
    <Table
      rowKey={(record) => `${record.name}:${record.path}`}
      columns={columns}
      dataSource={data}
      pagination={false}
      scroll={{ x: 1220 }}
      size="small"
    />
  );
}
