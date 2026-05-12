import ReactECharts from 'echarts-for-react';
import type { TrendPoint } from '../types';

interface MetricTrendChartProps {
  data: TrendPoint[];
}

export default function MetricTrendChart({ data }: MetricTrendChartProps) {
  const option = {
    grid: { left: 42, right: 42, top: 42, bottom: 36 },
    tooltip: { trigger: 'axis' },
    legend: { top: 8, data: ['loss', 'AP'] },
    xAxis: {
      type: 'category',
      data: data.map((item) => `step ${item.step}`),
      axisLabel: { color: '#667085' },
      axisLine: { lineStyle: { color: '#d9e2ef' } },
    },
    yAxis: [
      {
        type: 'value',
        name: 'loss',
        min: 0,
        axisLabel: { color: '#667085' },
        splitLine: { lineStyle: { color: '#edf2f7' } },
      },
      {
        type: 'value',
        name: 'AP',
        min: 0.82,
        max: 0.98,
        axisLabel: { color: '#667085' },
        splitLine: { show: false },
      },
    ],
    series: [
      {
        name: 'loss',
        type: 'line',
        smooth: true,
        data: data.map((item) => item.loss),
        lineStyle: { color: '#1677ff', width: 3 },
        itemStyle: { color: '#1677ff' },
      },
      {
        name: 'AP',
        type: 'line',
        smooth: true,
        yAxisIndex: 1,
        data: data.map((item) => item.ap),
        lineStyle: { color: '#13c2c2', width: 3 },
        itemStyle: { color: '#13c2c2' },
      },
    ],
  };

  return <ReactECharts option={option} style={{ height: 320, width: '100%' }} />;
}
