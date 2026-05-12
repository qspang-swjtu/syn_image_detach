import { ArrowRightOutlined } from '@ant-design/icons';

const flowGroups = [
  ['base_index.csv', 'increment_manifest'],
  ['all_samples.csv'],
  ['train_base', 'train_hard', 'val', 'test_unseen', 'reviewed_pool'],
  ['train_stage1', 'train_stage2', 'train_stage3'],
];

export default function DatasetFlow() {
  return (
    <div className="dataset-flow">
      {flowGroups.map((group, index) => (
        <div className="flow-segment" key={group.join('|')}>
          <div className="flow-card-group">
            {group.map((item) => (
              <div className="flow-node" key={item}>
                {item}
              </div>
            ))}
          </div>
          {index < flowGroups.length - 1 ? <ArrowRightOutlined className="flow-arrow" /> : null}
        </div>
      ))}
    </div>
  );
}
