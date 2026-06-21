import { useEffect, useState } from "react";
import { api, type BrowseResult } from "../api";
import { useI18n } from "../i18n";

interface Props {
  selected: Set<string>;
  onToggleFile: (path: string) => void;
  onSelectFolder: (path: string) => void;
}

export default function FileBrowser({ selected, onToggleFile, onSelectFolder }: Props) {
  const { t } = useI18n();
  const [data, setData] = useState<BrowseResult | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = (path: string) => {
    setErr(null);
    api.browse(path).then(setData).catch((e) => setErr(e.message));
  };

  useEffect(() => load(""), []);

  if (err) return <p className="text-sm text-red-500">{err}</p>;
  if (!data) return <p className="text-sm opacity-60">…</p>;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <code className="truncate text-xs opacity-60">/{data.path}</code>
        <button
          onClick={() => onSelectFolder(data.path)}
          className="shrink-0 rounded bg-indigo-600 px-2 py-1 text-xs font-medium text-white hover:bg-indigo-500"
        >
          {t("selectFolder")}
        </button>
      </div>

      <ul className="max-h-72 overflow-auto rounded border border-slate-200 dark:border-slate-700">
        {data.parent !== null && (
          <li>
            <button
              onClick={() => load(data.parent!)}
              className="w-full px-3 py-1.5 text-left text-sm hover:bg-slate-100 dark:hover:bg-slate-800"
            >
              📁 {t("parentDir")}
            </button>
          </li>
        )}
        {data.entries.map((e) =>
          e.is_dir ? (
            <li key={e.path}>
              <button
                onClick={() => load(e.path)}
                className="w-full px-3 py-1.5 text-left text-sm hover:bg-slate-100 dark:hover:bg-slate-800"
              >
                📁 {e.name}
              </button>
            </li>
          ) : (
            <li key={e.path}>
              <label className="flex cursor-pointer items-center gap-2 px-3 py-1.5 text-sm hover:bg-slate-100 dark:hover:bg-slate-800">
                <input
                  type="checkbox"
                  checked={selected.has(e.path)}
                  onChange={() => onToggleFile(e.path)}
                />
                🖼️ {e.name}
              </label>
            </li>
          )
        )}
      </ul>
    </div>
  );
}
