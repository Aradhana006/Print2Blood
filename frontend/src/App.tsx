import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { ComingSoonPage } from './pages/ComingSoonPage'
import { DashboardPage } from './pages/DashboardPage'
import { ModelsPage } from './pages/ModelsPage'
import { PredictionsPage } from './pages/PredictionsPage'
import { PredictPage } from './pages/PredictPage'
import { ResearchPage } from './pages/ResearchPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<DashboardPage />} />
          <Route path="predict" element={<PredictPage />} />
          <Route path="predictions" element={<PredictionsPage />} />
          <Route path="models" element={<ModelsPage />} />
          <Route path="datasets" element={<ComingSoonPage title="Datasets" phase="Phase 2" />} />
          <Route path="experiments" element={<ComingSoonPage title="Experiments" phase="Phase 3" />} />
          <Route path="reports" element={<ComingSoonPage title="Reports" phase="Phase 5" />} />
          <Route path="research" element={<ResearchPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
