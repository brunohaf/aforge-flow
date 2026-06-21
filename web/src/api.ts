// Thin wrappers around the FastAPI backend.

export interface Options {
  categories: string[];
  rembg_models: string[];
  image_formats: string[];
  video_formats: string[];
  compression_levels: string[];
  defaults: JobForm & { mode: string };
}

export interface JobForm {
  remove_background: boolean;
  optimize_media: boolean;
  category: string;
  rembg_model: string | null;
  image_format: string;
  video_format: string;
  compression: string;
  use_gpu: boolean;
  workers: number;
  skip_existing: boolean;
  output_dir: string;
}

export interface Entry {
  name: string;
  path: string;
  is_dir: boolean;
  is_supported: boolean;
  size: number | null;
}

export interface BrowseResult {
  root: string;
  path: string;
  parent: string | null;
  entries: Entry[];
}

export interface ProgressEvent {
  type: "start" | "file_done" | "done" | "error";
  total?: number;
  index?: number;
  input_path?: string;
  output_path?: string | null;
  success?: boolean;
  error?: string | null;
  input_size?: number | null;
  output_size?: number | null;
  summary?: Summary;
}

export interface Summary {
  total_files: number;
  successful: number;
  failed: number;
  success_rate: string;
  total_input_size: number;
  total_output_size: number;
  total_compression: string;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error((detail as any).detail || res.statusText);
  }
  return res.json() as Promise<T>;
}

export const api = {
  options: () => fetch("/api/options").then(json<Options>),

  browse: (path = "") =>
    fetch(`/api/browse?path=${encodeURIComponent(path)}`).then(json<BrowseResult>),

  upload: (files: FileList | File[]) => {
    const form = new FormData();
    Array.from(files).forEach((f) => form.append("files", f));
    return fetch("/api/upload", { method: "POST", body: form }).then(
      json<{ targets: string[] }>
    );
  },

  createJob: (body: Record<string, unknown>) =>
    fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(json<{ job_id: string }>),

  events: (jobId: string) => new EventSource(`/api/jobs/${jobId}/events`),

  fileUrl: (path: string, thumb = false) =>
    `/api/file?path=${encodeURIComponent(path)}${thumb ? "&thumb=true" : ""}`,

  downloadUrl: (path: string) => `/api/download?path=${encodeURIComponent(path)}`,

  zipUrl: (jobId: string) => `/api/jobs/${jobId}/download.zip`,
};
