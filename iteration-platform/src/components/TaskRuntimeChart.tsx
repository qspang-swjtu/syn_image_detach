import ReactECharts from 'echarts-for-react';
import type { TaskMetricPoint } from '../types/task';

interface TaskRuntimeChartProps {
  points: TaskMetricPoint[];
}

export default function TaskRuntimeChart({ points }: TaskRuntimeChartProps) {
  const option = {
    grid: { left: 46, right: 52, top: 42, bottom: 36 },
    tooltip: { trigger: 'axis' },
    legend: { top: 8, data: ['trainLoss', 'valAp', 'recallP98', 'lr'] },
    xAxis: {
      type: 'category',
      data: points.map((point) => `step ${point.step}`),
      axisLabel: { color: '#667085' },
      axisLine: { lineStyle: { color: '#d9e2ef' } },
    },
    yAxis: [
      {
        type: 'value',
        min: 0,
        axisLabel: { color: '#667085' },
        splitLine: { lineStyle: { color: '#edf2f7' } },
      },
    ],
    series: [
      { name: 'trainLoss', type: 'line', smooth: true, data: points.map((point) => point.trainLoss ?? null) },
      { name: 'valAp', type: 'line', smooth: true, data: points.map((point) => point.valAp ?? null) },
      { name: 'recallP98', type: 'line', smooth: true, data: points.map((point) => point.recallP98 ?? null) },
      { name: 'lr', type: 'line', smooth: true, data: points.map((point) => point.lr ?? null) },
    ],
  };

  return <ReactECharts option={option} style={{ width: '100%', height: 320 }} />;
}
