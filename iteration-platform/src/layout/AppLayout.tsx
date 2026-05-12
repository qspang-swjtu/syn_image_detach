import {
  BarChartOutlined,
  CheckCircleOutlined,
  DashboardOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  FolderOpenOutlined,
  PlusCircleOutlined,
  SettingOutlined,
} from '@ant-design/icons';
import { Breadcrumb, Layout, Menu, Space, Typography } from 'antd';
import type { MenuProps } from 'antd';
import { Outlet, useLocation, useNavigate, useParams } from 'react-router-dom';

const { Header, Sider, Content } = Layout;

function selectedKey(pathname: string) {
  if (pathname.startsWith('/dashboard')) return 'dashboard';
  if (pathname.startsWith('/dataset')) return 'dataset';
  if (pathname.startsWith('/tasks')) return 'monitor';
  if (pathname.startsWith('/evaluation')) return 'evaluation';
  if (pathname.startsWith('/compare')) return 'compare';
  if (pathname.startsWith('/models')) return 'models';
  if (pathname.startsWith('/settings')) return 'settings';
  return 'new';
}

const breadcrumbLabels: Record<string, string> = {
  dashboard: 'Dashboard 首页',
  dataset: '数据集管理',
  new: '新建迭代任务',
  monitor: '任务监控',
  evaluation: '评估结果',
  compare: '新旧模型对比',
  models: '模型仓库',
  settings: '系统配置',
};

export default function AppLayout() {
  const location = useLocation();
  const navigate = useNavigate();
  const params = useParams();
  const activeTaskId =
    params.taskId || localStorage.getItem('safepp:lastTaskId') || 'iter_20260511_001';
  const activeKey = selectedKey(location.pathname);

  const items: MenuProps['items'] = [
    {
      key: 'dashboard',
      icon: <DashboardOutlined />,
      label: 'Dashboard 首页',
      onClick: () => navigate('/dashboard'),
    },
    {
      key: 'dataset',
      icon: <DatabaseOutlined />,
      label: '数据集管理',
      onClick: () => navigate('/dataset'),
    },
    {
      key: 'new',
      icon: <PlusCircleOutlined />,
      label: '新建迭代任务',
      onClick: () => navigate('/iterations/create'),
    },
    {
      key: 'monitor',
      icon: <ExperimentOutlined />,
      label: '任务监控',
      onClick: () => navigate(`/tasks/${activeTaskId}`),
    },
    {
      key: 'evaluation',
      icon: <BarChartOutlined />,
      label: '评估结果',
      onClick: () => navigate(`/evaluation/${activeTaskId}`),
    },
    {
      key: 'compare',
      icon: <CheckCircleOutlined />,
      label: '新旧模型对比',
      onClick: () => navigate(`/compare/${activeTaskId}`),
    },
    {
      key: 'models',
      icon: <FolderOpenOutlined />,
      label: '模型仓库',
      onClick: () => navigate('/models'),
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: '系统配置',
      onClick: () => navigate('/settings'),
    },
  ];

  const currentBreadcrumb =
    activeKey === 'monitor' || activeKey === 'evaluation' || activeKey === 'compare'
      ? `${breadcrumbLabels[activeKey]} / ${activeTaskId}`
      : breadcrumbLabels[activeKey];

  return (
    <Layout className="app-shell">
      <Sider width={248} className="app-sider">
        <div className="brand">
          <ExperimentOutlined className="brand-icon" />
          <div>
            <Typography.Title level={4} className="brand-title">
              SafePP Iteration
            </Typography.Title>
            <Typography.Text type="secondary">合成图像鉴定模型平台</Typography.Text>
          </div>
        </div>
        <Menu mode="inline" selectedKeys={[activeKey]} items={items} className="side-menu" />
      </Sider>
      <Layout>
        <Header className="app-header">
          <Space direction="vertical" size={0}>
            <Typography.Title level={3} className="page-heading">
              合成图像鉴定模型自动化迭代平台
            </Typography.Title>
            <Breadcrumb items={[{ title: '自动化迭代' }, { title: currentBreadcrumb }]} />
          </Space>
        </Header>
        <Content className="app-content">
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  );
}
