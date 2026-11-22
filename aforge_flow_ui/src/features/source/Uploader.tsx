import { useRef, useState } from "react";
import { api } from "../../lib/api";
import { useI18n } from "../../i18n";

interface Props {
  onUploaded: (paths: string[]) => void;
  onRemove: (path: string) => void;
  selectedFiles: Set<string>;
}

function basename(p: string): string {
  return p.split(/[\\/]/).pop() || p;
}

export default function Uploader({ onUploaded, onRemove, selectedFiles }: Props) {
  const { t } = useI18n();
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const send = async (files: FileList | File[]) => {
    if (!files || (files as FileList).length === 0) return;
    setBusy(true);
    setErr(null);
    try {
      const res = await api.upload(files);
      onUploaded(res.targets);
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  };

  const uploadedPaths = Array.from(selectedFiles);

  return (
    <div className="space-y-2">
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          send(e.dataTransfer.files);
        }}
        className="cursor-pointer rounded border-2 border-dashed border-slate-300 p-6 text-center text-sm opacity-80 hover:border-indigo-500 dark:border-slate-600"
      >
        {busy ? "…" : t("dropHere")}
      </div>
      <input
        ref={inputRef}
        type="file"
        multiple
        className="hidden"
        onChange={(e) => e.target.files && send(e.target.files)}
      />
      {err && <p className="text-sm text-red-500">{err}</p>}
      {uploadedPaths.length > 0 && (
        <ul className="max-h-40 overflow-auto rounded border border-slate-200 dark:border-slate-700">
          {uploadedPaths.map((p) => (
            <li
              key={p}
              className="flex items-center justify-between gap-2 px-3 py-1.5 text-sm hover:bg-slate-50 dark:hover:bg-slate-800"
            >
              <span className="truncate">📄 {basename(p)}</span>
              <button
                onClick={() => onRemove(p)}
                className="shrink-0 text-slate-400 hover:text-red-500"
                title={t("remove")}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
