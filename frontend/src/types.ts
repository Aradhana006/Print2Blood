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

export type AnalyzeResponse = {
  request_id: string
  prediction: string
  confidence: number
  confidence_band: { level: string; meaning: string }
  probabilities: Record<string, number>
  top3: { label: string; probability: number }[]
  model_name: string
  model_version: string
  model_id: string
  timestamp: string
  image_hash: string
  quality: {
    width: number
    height: number
    brightness: number
    contrast: number
    sharpness: number
    scores: {
      overall: number
      sharpness: number
      contrast: number
      brightness: number
      ridge_activity: number
    }
    verdict: string
    tip: string
    pattern: {
      pattern_estimate: string
      orientation_peak: number
      orientation_entropy: number
      orientation_coherence: number
      note: string
    }
  }
  explainability: {
    method: string
    heatmap_overlay_jpeg_base64: string
    enhanced_fingerprint_jpeg_base64: string
    interpretation: string
  }
  ai_insights: string[]
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
