import {
  Alert,
  Button,
  Card,
  Col,
  Form,
  Input,
  InputNumber,
  Row,
  Select,
  Space,
  Switch,
  Typography,
  message,
} from 'antd';
import { CopyOutlined, DeleteOutlined, PlusOutlined, SaveOutlined } from '@ant-design/icons';
import { useMemo, useState } from 'react';
import { saveIncrementManifest } from '../../services/dataset';
import type { ManifestSource } from '../../types';

type ManifestSourceDraft = Partial<Omit<ManifestSource, 'label' | 'split_hint' | 'is_hard_negative'>> & {
  label?: 0 | 1;
  split_hint?: string;
  is_hard_negative?: 0 | 1;
};

type ManifestBuilderFormValues = {
  sources: ManifestSourceDraft[];
};

type ManifestBuilderProps = {
  iterationId: string;
  value?: string;
  onManifestSaved: (manifestPath: string) => void;
};

const defaultSource: ManifestSourceDraft = {
  name: 'hard_flux_20260511',
  path: '',
  label: 1,
  dataset: 'collected_hard',
  domain: 'fake',
  generator: 'flux',
  split_hint: 'hard',
  sample_weight: 3.0,
  is_hard_negative: 1,
  recursive: true,
};

function stringifyScalar(value: string | number | boolean): string {
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'number') return String(value);
  if (/^[A-Za-z0-9_./:\\-]+$/.test(value)) return value;
  return JSON.stringify(value);
}

function normalizeSource(source: ManifestSourceDraft): ManifestSource {
  const label = source.label ?? 1;
  return {
    name: source.name || 'new_source',
    path: source.path || '',
    label,
    source: source.source || source.name || 'new_source',
    dataset: source.dataset || 'unknown',
    domain: source.domain || (label === 0 ? 'real' : 'fake'),
    generator: source.generator || (label === 0 ? 'real' : 'unknown'),
    split_hint: source.split_hint || 'seen',
    sample_weight: source.sample_weight ?? 1.0,
    is_hard_negative: source.is_hard_negative ?? 0,
    recursive: source.recursive ?? true,
  };
}

function buildYaml(sources: ManifestSourceDraft[]): string {
  if (!sources.length) return 'sources: []\n';
  const lines = ['sources:'];
  sources.map(normalizeSource).forEach((source) => {
    lines.push(`  - name: ${stringifyScalar(source.name)}`);
    lines.push(`    path: ${stringifyScalar(source.path)}`);
    lines.push(`    label: ${source.label}`);
    lines.push(`    source: ${stringifyScalar(source.source || source.name)}`);
    lines.push(`    dataset: ${stringifyScalar(source.dataset || 'unknown')}`);
    lines.push(`    domain: ${stringifyScalar(source.domain || (source.label === 0 ? 'real' : 'fake'))}`);
    lines.push(`    generator: ${stringifyScalar(source.generator || (source.label === 0 ? 'real' : 'unknown'))}`);
    lines.push(`    split_hint: ${stringifyScalar(source.split_hint)}`);
    lines.push(`    sample_weight: ${source.sample_weight ?? 1.0}`);
    lines.push(`    is_hard_negative: ${source.is_hard_negative ?? 0}`);
    lines.push(`    recursive: ${stringifyScalar(source.recursive ?? true)}`);
  });
  return `${lines.join('\n')}\n`;
}

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : '保存 Manifest 失败';
}

export default function ManifestBuilder({ iterationId, value, onManifestSaved }: ManifestBuilderProps) {
  const [form] = Form.useForm<ManifestBuilderFormValues>();
  const [saving, setSaving] = useState(false);
  const [savedPath, setSavedPath] = useState<string | undefined>(value);
  const [warnings, setWarnings] = useState<string[]>([]);
  const sources = Form.useWatch('sources', form) || [defaultSource];
  const yamlText = useMemo(() => buildYaml(sources), [sources]);

  const applySplitDefaults = (index: number, splitHint: string) => {
    if (splitHint === 'hard') {
      form.setFieldValue(['sources', index, 'sample_weight'], 3.0);
      form.setFieldValue(['sources', index, 'is_hard_negative'], 1);
    }
    if (splitHint === 'unseen') {
      form.setFieldValue(['sources', index, 'sample_weight'], 1.0);
      form.setFieldValue(['sources', index, 'is_hard_negative'], 0);
    }
  };

  const applyLabelDefaults = (index: number, label: 0 | 1) => {
    if (label === 0) {
      form.setFieldValue(['sources', index, 'generator'], 'real');
      form.setFieldValue(['sources', index, 'domain'], 'real');
    } else {
      const currentGenerator = form.getFieldValue(['sources', index, 'generator']) as string | undefined;
      const currentDomain = form.getFieldValue(['sources', index, 'domain']) as string | undefined;
      if (!currentGenerator || currentGenerator === 'real') {
        form.setFieldValue(['sources', index, 'generator'], 'unknown');
      }
      if (!currentDomain || currentDomain === 'real') {
        form.setFieldValue(['sources', index, 'domain'], 'fake');
      }
    }
  };

  const handleSave = async () => {
    if (!iterationId.trim()) {
      message.warning('请先填写 iteration_id');
      return;
    }
    setSaving(true);
    setWarnings([]);
    try {
      const values = await form.validateFields();
      const normalizedSources = values.sources.map(normalizeSource);
      const response = await saveIncrementManifest({
        iterationId,
        sources: normalizedSources,
      });
      setSavedPath(response.manifestPath);
      setWarnings(response.warnings);
      onManifestSaved(response.manifestPath);
      message.success('Manifest 已保存');
    } catch (error) {
      message.error(getErrorMessage(error));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Space direction="vertical" size={14} className="page-stack manifest-builder">
      <Form
        form={form}
        layout="vertical"
        initialValues={{ sources: [defaultSource] }}
        component={false}
      >
        <Form.List name="sources">
          {(fields, { add, remove }) => (
            <Space direction="vertical" size={12} className="page-stack">
              {fields.map((field, index) => (
                <Card
                  size="small"
                  title={`Source ${index + 1}`}
                  className="manifest-source-card"
                  key={field.key}
                  extra={
                    <Space>
                      <Button
                        size="small"
                        icon={<CopyOutlined />}
                        onClick={() => {
                          const current = form.getFieldValue(['sources', field.name]) as ManifestSourceDraft;
                          add(
                            {
                              ...current,
                              name: `${current?.name || 'source'}_copy`,
                            },
                            field.name + 1,
                          );
                        }}
                      >
                        复制
                      </Button>
                      <Button
                        size="small"
                        danger
                        icon={<DeleteOutlined />}
                        disabled={fields.length === 1}
                        onClick={() => remove(field.name)}
                      >
                        删除
                      </Button>
                    </Space>
                  }
                >
                  <Row gutter={14}>
                    <Col xs={24} md={8}>
                      <Form.Item
                        name={[field.name, 'name']}
                        label="name"
                        rules={[{ required: true, message: '请输入 source name' }]}
                      >
                        <Input />
                      </Form.Item>
                    </Col>
                    <Col xs={24} md={16}>
                      <Form.Item
                        name={[field.name, 'path']}
                        label="path"
                        rules={[{ required: true, message: '请输入新增图片目录' }]}
                      >
                        <Input />
                      </Form.Item>
                    </Col>
                    <Col xs={24} md={6}>
                      <Form.Item
                        name={[field.name, 'label']}
                        label="label"
                        rules={[{ required: true, message: '请选择 label' }]}
                      >
                        <Select
                          options={[
                            { value: 0, label: '0 = 真实图' },
                            { value: 1, label: '1 = 合成图' },
                          ]}
                          onChange={(nextLabel: 0 | 1) => applyLabelDefaults(field.name, nextLabel)}
                        />
                      </Form.Item>
                    </Col>
                    <Col xs={24} md={6}>
                      <Form.Item name={[field.name, 'split_hint']} label="split_hint" rules={[{ required: true }]}>
                        <Select
                          options={[
                            { value: 'seen', label: 'seen' },
                            { value: 'hard', label: 'hard' },
                            { value: 'unseen', label: 'unseen' },
                            { value: 'reviewed_pool', label: 'reviewed_pool' },
                          ]}
                          onChange={(nextHint: string) => applySplitDefaults(field.name, nextHint)}
                        />
                      </Form.Item>
                    </Col>
                    <Col xs={24} md={6}>
                      <Form.Item name={[field.name, 'sample_weight']} label="sample_weight">
                        <InputNumber min={0} step={0.1} precision={2} className="full-width" />
                      </Form.Item>
                    </Col>
                    <Col xs={24} md={6}>
                      <Form.Item name={[field.name, 'is_hard_negative']} label="is_hard_negative">
                        <Select
                          options={[
                            { value: 0, label: '0' },
                            { value: 1, label: '1' },
                          ]}
                        />
                      </Form.Item>
                    </Col>
                    <Col xs={24} md={6}>
                      <Form.Item name={[field.name, 'source']} label="source">
                        <Input placeholder="默认等于 name" />
                      </Form.Item>
                    </Col>
                    <Col xs={24} md={6}>
                      <Form.Item name={[field.name, 'dataset']} label="dataset">
                        <Input placeholder="unknown" />
                      </Form.Item>
                    </Col>
                    <Col xs={24} md={6}>
                      <Form.Item name={[field.name, 'domain']} label="domain">
                        <Input placeholder="fake / real" />
                      </Form.Item>
                    </Col>
                    <Col xs={24} md={6}>
                      <Form.Item name={[field.name, 'generator']} label="generator">
                        <Input placeholder="real / flux / sdxl" />
                      </Form.Item>
                    </Col>
                    <Col xs={24} md={6}>
                      <Form.Item name={[field.name, 'recursive']} label="recursive" valuePropName="checked">
                        <Switch checkedChildren="true" unCheckedChildren="false" />
                      </Form.Item>
                    </Col>
                  </Row>
                </Card>
              ))}
              <Button icon={<PlusOutlined />} onClick={() => add(defaultSource)} block>
                添加 source
              </Button>
            </Space>
          )}
        </Form.List>
      </Form>

      <Card size="small" title="YAML 预览" className="yaml-preview-card">
        <pre className="yaml-preview">{yamlText}</pre>
      </Card>

      {warnings.length > 0 ? <Alert type="warning" showIcon message={warnings.join('；')} /> : null}
      {savedPath ? (
        <Alert
          type="success"
          showIcon
          message="Manifest 已保存"
          description={
            <Typography.Text copyable className="path-value">
              {savedPath}
            </Typography.Text>
          }
        />
      ) : null}

      <div className="form-actions">
        <Button type="primary" icon={<SaveOutlined />} loading={saving} onClick={handleSave}>
          保存 Manifest
        </Button>
      </div>
    </Space>
  );
}
