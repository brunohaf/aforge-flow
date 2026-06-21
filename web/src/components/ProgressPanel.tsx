import { useEffect, useState } from "react";
import { api, type ProgressEvent, type Summary } from "../api";
import { useI18n } from "../i18n";

interface Props {
  jobId: string;
}

interface FileRow {
  input_path: string;
  output_path: string | null;
  success: boolean;
  error: string | null;
}

export default function ProgressPanel({ jobId }: Props) {
  const { t } = useI18n();
  const [total, setTotal] = useState(0);
  const [rows, setRows] = useState<FileRow[]>([]);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Reset when a new job starts.
    setTotal(0);
    setRows([]);
    setSummary(null);
    setError(null);

    const es = api.events(jobId);
    es.onmessage = (msg) => {
      const ev: ProgressEvent = JSON.parse(msg.data);
      if (ev.type === "start") setTotal(ev.total ?? 0);
      else if (ev.type === "file_done")
        setRows((r) => [
          ...r,
          {
            input_path: ev.input_path!,
            output_path: ev.output_path ?? null,
            success: !!ev.success,
            error: ev.error ?? null,
          },
        ]);
      else if (ev.type === "done") {
        setSummary(ev.summary ?? null);
        es.close();
      } else if (ev.type === "error") {
        setError(ev.error ?? "error");
        es.close();
      }
    };
    es.onerror = () => es.close();
    return () => es.close();
  }, [jobId]);

  const pct = total > 0 ? Math.round((rows.length / total) * 100) : 0;
  const hasOutputs = rows.some((r) => r.success && r.output_path);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="font-semibold">{t("progress")}</h2>
        {summary && hasOutputs && (
          <a
            href={api.zipUrl(jobId)}
            className="rounded bg-emerald-600 px-3 py-1 text-sm font-medium text-white hover:bg-emerald-500"
          >
            {t("downloadAll")}
          </a>
        )}
      </div>

      <div className="h-2 w-full overflow-hidden rounded bg-slate-200 dark:bg-slate-700">
        <div
          className="h-full bg-indigo-600 transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="text-xs opacity-60">
        {total === 0 && !summary ? t("waiting") : `${rows.length} / ${total} · ${pct}%`}
      </p>

      {error && <p className="text-sm text-red-500">{t("error")}: {error}</p>}

      <ul className="space-y-2">
        {rows.map((r, i) => (
          <li
            key={i}
            className="rounded border border-slate-200 p-2 dark:border-slate-700"
          >
            <div className="mb-1 flex items-center justify-between gap-2">
              <span className="truncate text-sm">
                {r.success ? "✅" : "❌"} {basename(r.input_path)}
              </span>
              {r.success && r.output_path && (
                <a
                  href={api.downloadUrl(r.output_path)}
                  className="shrink-0 text-xs font-medium text-indigo-600 hover:underline dark:text-indigo-400"
                >
                  ⬇ {t("download")}
                </a>
              )}
            </div>
            {r.success ? (
              <div className="flex gap-3">
                <Thumb label={t("before")} path={r.input_path} />
                {r.output_path && <Thumb label={t("after")} path={r.output_path} />}
              </div>
            ) : (
              <p className="text-xs text-red-500">{r.error}</p>
            )}
          </li>
        ))}
      </ul>

      {summary && (
        <div className="rounded bg-slate-100 p-3 text-sm dark:bg-slate-800">
          <p>{t("done")} · {summary.successful}/{summary.total_files}</p>
          <p className="opacity-70">
            {t("successRate")}: {summary.success_rate} · {t("compressionAchieved")}:{" "}
            {summary.total_compression}
          </p>
        </div>
      )}
    </div>
  );
}

function Thumb({ label, path }: { label: string; path: string }) {
  return (
    <figure className="text-center">
      <img
        src={api.fileUrl(path, true)}
        alt={label}
        className="h-24 w-24 rounded border border-slate-200 object-contain dark:border-slate-700"
        style={{
          backgroundImage:
            "repeating-conic-gradient(#cbd5e1 0% 25%, transparent 0% 50%)",
          backgroundSize: "16px 16px",
        }}
      />
      <figcaption className="text-[10px] opacity-60">{label}</figcaption>
    </figure>
  );
}

function basename(p: string): string {
  return p.split(/[\\/]/).pop() || p;
}
