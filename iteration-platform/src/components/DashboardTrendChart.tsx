import ReactECharts from 'echarts-for-react';
import type { MetricTrendRecord } from '../types';

interface DashboardTrendChartProps {
  data: MetricTrendRecord[];
}

export default function DashboardTrendChart({ data }: DashboardTrendChartProps) {
  const option = {
    grid: { left: 46, right: 52, top: 48, bottom: 38 },
    tooltip: { trigger: 'axis' },
    legend: {
      top: 8,
      data: ['AP', 'Recall@P98', 'AUROC', 'Real FPR', 'Fake FNR'],
    },
    xAxis: {
      type: 'category',
      data: data.map((item) => item.version),
      axisLabel: { color: '#667085' },
      axisLine: { lineStyle: { color: '#d9e2ef' } },
    },
    yAxis: [
      {
        type: 'value',
        min: 0,
        max: 1,
        axisLabel: { color: '#667085' },
        splitLine: { lineStyle: { color: '#edf2f7' } },
      },
    ],
    series: [
      { name: 'AP', type: 'line', smooth: true, data: data.map((item) => item.ap) },
      { name: 'Recall@P98', type: 'line', smooth: true, data: data.map((item) => item.recall_p98) },
      { name: 'AUROC', type: 'line', smooth: true, data: data.map((item) => item.auroc) },
      { name: 'Real FPR', type: 'line', smooth: true, data: data.map((item) => item.real_fpr) },
      { name: 'Fake FNR', type: 'line', smooth: true, data: data.map((item) => item.fake_fnr) },
    ],
  };

  return <ReactECharts option={option} style={{ width: '100%', height: 340 }} />;
}
