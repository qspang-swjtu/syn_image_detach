import { Tree } from 'antd';
import type { DataNode } from 'antd/es/tree';

const treeData: DataNode[] = [
  {
    title: 'model',
    key: 'model',
    children: [{ title: 'best.pt', key: 'model/best.pt' }],
  },
  {
    title: 'config',
    key: 'config',
    children: [
      { title: 'stage1.yaml', key: 'config/stage1.yaml' },
      { title: 'stage2.yaml', key: 'config/stage2.yaml' },
    ],
  },
  {
    title: 'metrics',
    key: 'metrics',
    children: [
      { title: 'val.json', key: 'metrics/val.json' },
      { title: 'test_unseen.json', key: 'metrics/test_unseen.json' },
      { title: 'test_all.json', key: 'metrics/test_all.json' },
    ],
  },
  {
    title: 'threshold',
    key: 'threshold',
    children: [{ title: 'thresholds.json', key: 'threshold/thresholds.json' }],
  },
  { title: 'manifest.json', key: 'manifest.json' },
];

export default function ModelArtifactTree() {
  return <Tree treeData={treeData} defaultExpandAll blockNode />;
}
