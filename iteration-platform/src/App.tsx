import { Navigate, Route, Routes } from 'react-router-dom';
import { BrowserRouter } from 'react-router-dom';
import AppLayout from './layout/AppLayout';
import DashboardPage from './pages/DashboardPage';
import DatasetPage from './pages/DatasetPage';
import NewIterationPage from './pages/NewIterationPage';
import TaskMonitorPage from './pages/TaskMonitorPage';
import EvaluationPage from './pages/EvaluationPage';
import ModelComparePage from './pages/ModelComparePage';
import ModelsPage from './pages/ModelsPage';
import SettingsPage from './pages/SettingsPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/dataset" element={<DatasetPage />} />
          <Route path="/iterations/create" element={<NewIterationPage />} />
          <Route path="/iterations/new" element={<NewIterationPage />} />
          <Route path="/tasks/:taskId" element={<TaskMonitorPage />} />
          <Route path="/evaluation/:taskId" element={<EvaluationPage />} />
          <Route path="/compare/:taskId" element={<ModelComparePage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
