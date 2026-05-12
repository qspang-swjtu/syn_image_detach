import { Card } from 'antd';
import type { ReactNode } from 'react';

interface SettingsFormSectionProps {
  title: string;
  children: ReactNode;
}

export default function SettingsFormSection({ title, children }: SettingsFormSectionProps) {
  return (
    <Card title={title} className="panel-card settings-section">
      {children}
    </Card>
  );
}
