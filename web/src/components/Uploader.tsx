import { useRef, useState } from "react";
import { api } from "../api";
import { useI18n } from "../i18n";

interface Props {
  onUploaded: (paths: string[]) => void;
}

export default function Uploader({ onUploaded }: Props) {
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
    </div>
  );
}
