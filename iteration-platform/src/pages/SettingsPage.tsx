import { Button, Col, Form, Input, InputNumber, Radio, Row, Space, Spin, Switch, Typography, message } from 'antd';
import { SaveOutlined } from '@ant-design/icons';
import { useEffect, useState } from 'react';
import GateRuleForm from '../components/GateRuleForm';
import SettingsFormSection from '../components/SettingsFormSection';
import { getSettings, saveSettings } from '../services/settings';
import type { PlatformSettings } from '../types';

const stageSwitches: Array<{ name: keyof PlatformSettings; label: string }> = [
  { name: 'run_stage1', label: 'run_stage1' },
  { name: 'run_stage2', label: 'run_stage2' },
  { name: 'run_replay', label: 'run_replay' },
  { name: 'run_stage3', label: 'run_stage3' },
  { name: 'run_eval', label: 'run_eval' },
];

export default function SettingsPage() {
  const [form] = Form.useForm<PlatformSettings>();
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getSettings().then((settings) => {
      form.setFieldsValue(settings);
      setLoading(false);
    });
  }, [form]);

  const handleFinish = (values: PlatformSettings) => {
    saveSettings(values).then(() => {
      message.success('配置已保存');
    });
  };

  if (loading) {
    return (
      <div className="center-box">
        <Spin tip="加载系统配置" />
      </div>
    );
  }

  return (
    <Space direction="vertical" size={18} className="page-stack">
      <div className="page-intro">
        <div>
          <Typography.Title level={2}>系统配置</Typography.Title>
          <Typography.Paragraph type="secondary">
            配置平台默认路径、训练参数和评估 gate 规则。
          </Typography.Paragraph>
        </div>
      </div>

      <Form form={form} layout="vertical" onFinish={handleFinish}>
        <SettingsFormSection title="默认路径配置">
          <Row gutter={16}>
            <Col xs={24} lg={12}>
              <Form.Item name="default_base_csv" label="default_base_csv">
                <Input />
              </Form.Item>
            </Col>
            <Col xs={24} lg={12}>
              <Form.Item name="default_increment_manifest" label="default_increment_manifest">
                <Input />
              </Form.Item>
            </Col>
            <Col xs={24} lg={8}>
              <Form.Item name="default_output_dir" label="default_output_dir">
                <Input />
              </Form.Item>
            </Col>
            <Col xs={24} lg={8}>
              <Form.Item name="default_model_registry_dir" label="default_model_registry_dir">
                <Input />
              </Form.Item>
            </Col>
            <Col xs={24} lg={8}>
              <Form.Item name="default_log_dir" label="default_log_dir">
                <Input />
              </Form.Item>
            </Col>
          </Row>
        </SettingsFormSection>

        <SettingsFormSection title="训练默认参数">
          <Row gutter={16}>
            <Col xs={24} md={8}>
              <Form.Item name="default_seed" label="default_seed">
                <InputNumber min={0} precision={0} className="full-width" />
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item name="default_nproc" label="default_nproc">
                <InputNumber min={1} max={128} precision={0} className="full-width" />
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item name="default_train_plan" label="default_train_plan">
                <Radio.Group>
                  <Radio.Button value="hard_in_stage1">hard_in_stage1</Radio.Button>
                  <Radio.Button value="hard_in_stage2">hard_in_stage2</Radio.Button>
                </Radio.Group>
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item name="default_val_real_total" label="default_val_real_total">
                <InputNumber min={1} precision={0} className="full-width" />
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item name="default_val_fake_total" label="default_val_fake_total">
                <InputNumber min={1} precision={0} className="full-width" />
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={16}>
            {stageSwitches.map((item) => (
              <Col xs={12} md={4} key={item.name}>
                <Form.Item name={item.name} label={item.label} valuePropName="checked">
                  <Switch checkedChildren="on" unCheckedChildren="off" />
                </Form.Item>
              </Col>
            ))}
          </Row>
        </SettingsFormSection>

        <SettingsFormSection title="Gate 规则配置">
          <GateRuleForm />
        </SettingsFormSection>

        <div className="form-actions">
          <Button type="primary" htmlType="submit" size="large" icon={<SaveOutlined />}>
            保存配置
          </Button>
        </div>
      </Form>
    </Space>
  );
}
