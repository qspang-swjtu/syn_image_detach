import { Col, Form, InputNumber, Row, Switch } from 'antd';

export default function GateRuleForm() {
  return (
    <Row gutter={16}>
      <Col xs={24} md={8}>
        <Form.Item
          name="val_ap_min_delta"
          label="val_ap_min_delta"
          tooltip="候选模型相对 baseline 的 AP 最小变化"
        >
          <InputNumber step={0.001} precision={4} className="full-width" />
        </Form.Item>
      </Col>
      <Col xs={24} md={8}>
        <Form.Item
          name="val_recall_p98_min_delta"
          label="val_recall_p98_min_delta"
          tooltip="Recall@P98 允许的最小变化"
        >
          <InputNumber step={0.001} precision={4} className="full-width" />
        </Form.Item>
      </Col>
      <Col xs={24} md={8}>
        <Form.Item name="hard_recall_p98_min_delta" label="hard_recall_p98_min_delta">
          <InputNumber step={0.001} precision={4} className="full-width" />
        </Form.Item>
      </Col>
      <Col xs={24} md={8}>
        <Form.Item
          name="real_fpr_max_delta"
          label="real_fpr_max_delta"
          tooltip="真实图误杀率允许的最大上升幅度"
        >
          <InputNumber step={0.001} precision={4} className="full-width" />
        </Form.Item>
      </Col>
      <Col xs={24} md={8}>
        <Form.Item name="fake_fnr_max_delta" label="fake_fnr_max_delta">
          <InputNumber step={0.001} precision={4} className="full-width" />
        </Form.Item>
      </Col>
      <Col xs={24} md={8}>
        <Form.Item
          name="require_test_unseen"
          label="require_test_unseen"
          valuePropName="checked"
          tooltip="是否强制要求存在泛化测试集"
        >
          <Switch checkedChildren="required" unCheckedChildren="optional" />
        </Form.Item>
      </Col>
    </Row>
  );
}
