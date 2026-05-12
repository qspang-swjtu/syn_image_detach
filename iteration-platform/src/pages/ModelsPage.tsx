import {
  Alert,
  Button,
  Card,
  Descriptions,
  Drawer,
  Space,
  Spin,
  Table,
  Typography,
  message,
} from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ModelArtifactTree from '../components/ModelArtifactTree';
import ModelStatusTag from '../components/ModelStatusTag';
import { archiveModel, getModelDetail, getModelList, promoteModel } from '../services/model';
import type { ModelDetail, ModelRecord } from '../types';

const formatMetric = (value: number) => value.toFixed(4);

export default function ModelsPage() {
  const navigate = useNavigate();
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [selectedModel, setSelectedModel] = useState<ModelDetail>();
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getModelList().then((data) => {
      setModels(data);
      setLoading(false);
    });
  }, []);

  const openDetail = (modelId: string) => {
    getModelDetail(modelId).then(setSelectedModel);
  };

  const updateModelRow = (updated: ModelRecord) => {
    setModels((current) => current.map((item) => (item.model_id === updated.model_id ? updated : item)));
  };

  const columns: ColumnsType<ModelRecord> = [
    { title: 'model_id', dataIndex: 'model_id', fixed: 'left', width: 190 },
    { title: 'iteration_id', dataIndex: 'iteration_id', width: 160 },
    { title: 'train_plan', dataIndex: 'train_plan', width: 150 },
    {
      title: 'status',
      dataIndex: 'status',
      width: 120,
      render: (value: ModelRecord['status']) => <ModelStatusTag status={value} />,
    },
    { title: 'ap', dataIndex: 'ap', render: formatMetric },
    { title: 'auroc', dataIndex: 'auroc', render: formatMetric },
    { title: 'recall_p98', dataIndex: 'recall_p98', render: formatMetric },
    { title: 'real_fpr', dataIndex: 'real_fpr', render: formatMetric },
    { title: 'fake_fnr', dataIndex: 'fake_fnr', render: formatMetric },
    { title: 'created_at', dataIndex: 'created_at', width: 180 },
    {
      title: 'action',
      key: 'action',
      fixed: 'right',
      width: 420,
      render: (_, record) => (
        <Space wrap>
          <Button size="small" onClick={() => openDetail(record.model_id)}>
            查看详情
          </Button>
          <Button size="small" onClick={() => navigate(`/evaluation/${record.iteration_id}`)}>
            查看指标
          </Button>
          <Button size="small" onClick={() => message.success(`${record.model_id} checkpoint 下载已模拟触发`)}>
            下载 checkpoint
          </Button>
          <Button size="small" onClick={() => message.success(`${record.model_id} 已设为 baseline`)}>
            设为 baseline
          </Button>
          <Button
            size="small"
            type="primary"
            onClick={() =>
              promoteModel(record.model_id).then((updated) => {
                updateModelRow(updated);
                message.success(`${record.model_id} 已设为 production`);
              })
            }
          >
            设为 production
          </Button>
          <Button
            size="small"
            danger
            onClick={() =>
              archiveModel(record.model_id).then((updated) => {
                updateModelRow(updated);
                message.success(`${record.model_id} 已归档，可用于回滚`);
              })
            }
          >
            回滚
          </Button>
        </Space>
      ),
    },
  ];

  if (loading) {
    return (
      <div className="center-box">
        <Spin tip="加载模型仓库" />
      </div>
    );
  }

  return (
    <Space direction="vertical" size={18} className="page-stack">
      <div className="page-intro">
        <div>
          <Typography.Title level={2}>模型仓库</Typography.Title>
          <Typography.Paragraph type="secondary">
            管理 production、candidate、archived、failed 模型版本和训练产物。
          </Typography.Paragraph>
        </div>
      </div>

      <Card title="模型列表" className="panel-card">
        <Table
          rowKey="model_id"
          columns={columns}
          dataSource={models}
          pagination={false}
          scroll={{ x: 1620 }}
        />
      </Card>

      <Card title="模型产物结构" className="panel-card">
        <ModelArtifactTree />
      </Card>

      <Alert
        type="info"
        showIcon
        message="模型状态说明"
        description="production：当前线上使用模型；candidate：通过训练与评估，但尚未发布；archived：历史模型，可用于回滚；failed：训练或评估失败的模型。"
      />

      <Drawer title="模型详情" open={Boolean(selectedModel)} onClose={() => setSelectedModel(undefined)} width={620}>
        {selectedModel ? (
          <Descriptions column={1} bordered size="small">
            <Descriptions.Item label="model_id">{selectedModel.model_id}</Descriptions.Item>
            <Descriptions.Item label="iteration_id">{selectedModel.iteration_id}</Descriptions.Item>
            <Descriptions.Item label="train_plan">{selectedModel.train_plan}</Descriptions.Item>
            <Descriptions.Item label="checkpoint_path">{selectedModel.checkpoint_path}</Descriptions.Item>
            <Descriptions.Item label="config_path">{selectedModel.config_path}</Descriptions.Item>
            <Descriptions.Item label="metrics_dir">{selectedModel.metrics_dir}</Descriptions.Item>
            <Descriptions.Item label="threshold_path">{selectedModel.threshold_path}</Descriptions.Item>
            <Descriptions.Item label="data_version">{selectedModel.data_version}</Descriptions.Item>
            <Descriptions.Item label="git_commit">{selectedModel.git_commit}</Descriptions.Item>
            <Descriptions.Item label="created_at">{selectedModel.created_at}</Descriptions.Item>
          </Descriptions>
        ) : null}
      </Drawer>
    </Space>
  );
}
