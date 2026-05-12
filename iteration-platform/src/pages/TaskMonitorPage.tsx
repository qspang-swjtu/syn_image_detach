import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Progress,
  Row,
  Space,
  Spin,
  Steps,
  Switch,
  Tag,
  Typography,
  message,
} from 'antd';
import { BarChartOutlined, ReloadOutlined, RocketOutlined } from '@ant-design/icons';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import TaskRuntimeChart from '../components/TaskRuntimeChart';
import {
  getIterationTask,
  getTaskLogs,
  getTaskRuntimeMetrics,
  startIterationTask,
} from '../services/task';
import type {
  IterationStage,
  IterationTaskDetail,
  StageStatus,
  TaskLogLevel,
  TaskLogLine,
  TaskRuntimeMetrics,
  TaskStatus,
} from '../types/task';

const statusColor: Record<TaskStatus, string> = {
  created: 'default',
  pending: 'gold',
  running: 'processing',
  success: 'success',
  failed: 'error',
  stopped: 'default',
};

const logColor: Record<TaskLogLevel, string> = {
  INFO: '#344054',
  WARN: '#b54708',
  ERROR: '#b42318',
  DEBUG: '#667085',
};

function stepStatus(status: StageStatus): 'wait' | 'process' | 'finish' | 'error' {
  if (status === 'success' || status === 'skipped') return 'finish';
  if (status === 'running') return 'process';
  if (status === 'failed') return 'error';
  return 'wait';
}

function getErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : '请求失败';
}

function isPollingStatus(status: TaskStatus) {
  return status === 'running' || status === 'pending';
}

export default function TaskMonitorPage() {
  const { taskId = 'iter_20260511_001' } = useParams();
  const navigate = useNavigate();
  const [task, setTask] = useState<IterationTaskDetail>();
  const [logs, setLogs] = useState<TaskLogLine[]>([]);
  const [metrics, setMetrics] = useState<TaskRuntimeMetrics>({ iterationId: taskId, points: [] });
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [errorOnly, setErrorOnly] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [error, setError] = useState<string>();
  const logRef = useRef<HTMLDivElement>(null);

  const loadTask = useCallback(async () => {
    const detail = await getIterationTask(taskId);
    setTask(detail);
  }, [taskId]);

  const loadLogs = useCallback(async () => {
    const response = await getTaskLogs(taskId, { level: errorOnly ? 'ERROR' : undefined });
    setLogs(response.lines);
  }, [errorOnly, taskId]);

  const loadMetrics = useCallback(async () => {
    setMetrics(await getTaskRuntimeMetrics(taskId));
  }, [taskId]);

  const refreshAll = useCallback(async () => {
    setError(undefined);
    try {
      await Promise.all([loadTask(), loadLogs(), loadMetrics()]);
    } catch (err) {
      const msg = getErrorMessage(err);
      setError(msg);
      message.error(msg);
    } finally {
      setLoading(false);
    }
  }, [loadLogs, loadMetrics, loadTask]);

  useEffect(() => {
    setLoading(true);
    refreshAll();
  }, [refreshAll]);

  useEffect(() => {
    if (!task || !isPollingStatus(task.status)) return;
    const statusTimer = window.setInterval(() => {
      loadTask().catch((err) => setError(getErrorMessage(err)));
      loadLogs().catch((err) => setError(getErrorMessage(err)));
    }, 5000);
    const metricsTimer = window.setInterval(() => {
      loadMetrics().catch((err) => setError(getErrorMessage(err)));
    }, 10000);
    return () => {
      window.clearInterval(statusTimer);
      window.clearInterval(metricsTimer);
    };
  }, [loadLogs, loadMetrics, loadTask, task]);

  useEffect(() => {
    if (autoScroll && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [autoScroll, logs]);

  const steps = useMemo(
    () =>
      (task?.stages || []).map((stage: IterationStage) => ({
        title: stage.title,
        status: stepStatus(stage.status),
        description: stage.status === 'skipped' ? 'skipped' : stage.message,
      })),
    [task],
  );
  const currentStep = Math.max(0, steps.findIndex((item) => item.status === 'process'));

  const handleStart = async () => {
    if (task && !task.dataPrepared) {
      message.warning('数据尚未准备完成，请先在新建迭代任务页面完成启动前检查/创建任务。');
      return;
    }
    setStarting(true);
    setError(undefined);
    try {
      setTask(await startIterationTask(taskId));
      await Promise.all([loadLogs(), loadMetrics()]);
      message.success('任务已启动');
    } catch (err) {
      const msg = getErrorMessage(err);
      setError(msg);
      message.error(msg);
    } finally {
      setStarting(false);
    }
  };

  if (loading && !task) {
    return (
      <div className="center-box">
        <Spin tip="加载任务状态" />
      </div>
    );
  }

  return (
    <Space direction="vertical" size={18} className="page-stack">
      <div className="page-intro">
        <div>
          <Typography.Title level={2}>任务监控</Typography.Title>
          <Typography.Paragraph type="secondary">
            查看迭代任务状态、阶段进度、运行日志和训练过程指标。
          </Typography.Paragraph>
        </div>
      </div>

      {error ? <Alert type="error" showIcon message="接口调用失败" description={error} /> : null}
      {task && !task.dataPrepared ? (
        <Alert
          type="warning"
          showIcon
          message="数据尚未准备完成，请先在新建迭代任务页面完成启动前检查/创建任务。"
          description={
            task.missingPreparedFiles.length > 0
              ? `缺失文件：${task.missingPreparedFiles.join('，')}`
              : undefined
          }
        />
      ) : null}

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={16}>
          <Card className="panel-card">
            <div className="section-head">
              <div>
                <Typography.Title level={3}>{task?.iterationId || taskId}</Typography.Title>
                <Typography.Text type="secondary">{task?.description || '迭代任务详情'}</Typography.Text>
              </div>
              <Tag color={task ? statusColor[task.status] : 'default'} className="status-tag">
                {task?.status || 'unknown'}
              </Tag>
            </div>
            <Progress
              percent={task?.progress ?? 0}
              status={task?.status === 'failed' ? 'exception' : task?.status === 'success' ? 'success' : 'active'}
            />
            <Steps current={currentStep} items={steps} size="small" />
          </Card>
        </Col>
        <Col xs={24} xl={8}>
          <Card className="panel-card">
            <Descriptions column={1} size="small" labelStyle={{ width: 120 }}>
              <Descriptions.Item label="trainPlan">{task?.trainPlan || '-'}</Descriptions.Item>
              <Descriptions.Item label="dataPrepared">
                <Tag color={task?.dataPrepared ? 'green' : 'red'}>{task?.dataPrepared ? 'ready' : 'missing'}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="dataDir">{task?.dataDir || '-'}</Descriptions.Item>
              <Descriptions.Item label="outputDir">{task?.outputDir || '-'}</Descriptions.Item>
              <Descriptions.Item label="allSamplesCsv">{task?.allSamplesCsv || '-'}</Descriptions.Item>
              <Descriptions.Item label="trainStage1Csv">{task?.trainStage1Csv || '-'}</Descriptions.Item>
              <Descriptions.Item label="trainStage2Csv">{task?.trainStage2Csv || '-'}</Descriptions.Item>
              <Descriptions.Item label="valCsv">{task?.valCsv || '-'}</Descriptions.Item>
              <Descriptions.Item label="startedAt">{task?.startedAt || '-'}</Descriptions.Item>
              <Descriptions.Item label="elapsedSeconds">{task?.elapsedSeconds ?? '-'}</Descriptions.Item>
              <Descriptions.Item label="currentStage">{task?.currentStage || '-'}</Descriptions.Item>
            </Descriptions>
          </Card>
        </Col>
      </Row>

      <Space wrap>
        <Button
          type="primary"
          icon={<RocketOutlined />}
          loading={starting}
          disabled={Boolean(task && !task.dataPrepared)}
          onClick={handleStart}
        >
          启动任务
        </Button>
        <Button icon={<ReloadOutlined />} onClick={refreshAll}>
          刷新状态
        </Button>
        <Button icon={<BarChartOutlined />} onClick={() => navigate(`/evaluation/${taskId}`)}>
          查看评估结果
        </Button>
      </Space>

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={10}>
          <Card
            title="训练日志"
            className="panel-card"
            extra={
              <Space>
                <Typography.Text type="secondary">只看 ERROR</Typography.Text>
                <Switch checked={errorOnly} onChange={setErrorOnly} />
                <Typography.Text type="secondary">自动滚动</Typography.Text>
                <Switch checked={autoScroll} onChange={setAutoScroll} />
                <Button size="small" onClick={loadLogs}>
                  刷新日志
                </Button>
              </Space>
            }
          >
            <div className="task-log-window" ref={logRef}>
              {logs.length === 0 ? (
                <Empty description="暂无日志" />
              ) : (
                logs.map((line) => (
                  <div className="task-log-line" key={line.id} style={{ color: logColor[line.level] }}>
                    <span>[{line.timestamp}]</span>
                    <Tag color={line.level === 'ERROR' ? 'red' : line.level === 'WARN' ? 'orange' : 'blue'}>
                      {line.level}
                    </Tag>
                    {line.stage ? <span>{line.stage}</span> : null}
                    <span>{line.message}</span>
                  </div>
                ))
              )}
            </div>
          </Card>
        </Col>
        <Col xs={24} xl={14}>
          <Card title="实时指标" className="panel-card">
            {metrics.points.length === 0 ? (
              <Empty description="暂无可解析训练指标" />
            ) : (
              <TaskRuntimeChart points={metrics.points} />
            )}
          </Card>
        </Col>
      </Row>
    </Space>
  );
}
