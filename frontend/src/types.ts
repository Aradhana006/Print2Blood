export type PredictResponse = {
  request_id: string
  prediction: string
  confidence: number
  probabilities: Record<string, number>
  top3: { label: string; probability: number }[]
  model_name: string
  model_version: string
  model_id: string
  preprocessing_version: string
  image_hash: string
  timestamp: string
  disclaimer: string
}

export type DashboardSummary = {
  app_name: string
  total_dataset_images: number
  n_classes: number
  deployed_model_name: string | null
  deployed_model_version: string | null
  deployed_model_accuracy: number | null
  deployed_model_macro_f1: number | null
  total_predictions: number
  total_experiments: number
  leakage_status: string
  class_distribution: Record<string, number>
  disclaimer: string
}

export type ModelInfo = {
  id: string
  model_name: string
  version: string
  architecture: string
  status: string
  is_active: boolean
  accuracy: number | null
  macro_f1: number | null
  split_type: string | null
  image_size: number
  notes: string | null
}

export type PredictionHistoryItem = {
  request_id: string
  timestamp: string
  prediction: string
  confidence: number
  model_id: string
  image_hash: string
  original_filename: string | null
}
