import { useEffect, useMemo, useState, type ReactNode } from "react";
import { api, type JobForm, type Options } from "./api";
import FileBrowser from "./components/FileBrowser";
import Uploader from "./components/Uploader";
import OptionsForm from "./components/OptionsForm";
import ProgressPanel from "./components/ProgressPanel";
import { useI18n } from "./i18n";
import { LOCALES, type Locale } from "./locales";
import { useTheme } from "./theme";

type SourceTab = "browse" | "upload";

export default function App() {
  const { t, locale, setLocale } = useI18n();
  const { dark, toggle } = useTheme();

  const [options, setOptions] = useState<Options | null>(null);
  const [form, setForm] = useState<JobForm | null>(null);
  const [tab, setTab] = useState<SourceTab>("browse");

  // Selection state: either a folder (directory mode) or a set of files.
  const [folder, setFolder] = useState<string | null>(null);
  const [files, setFiles] = useState<Set<string>>(new Set());

  const [jobId, setJobId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.options().then((o) => {
      setOptions(o);
      const d = o.defaults as unknown as JobForm;
      setForm({ ...d });
    });
  }, []);

  const update = <K extends keyof JobForm>(key: K, value: JobForm[K]) =>
    setForm((f) => (f ? { ...f, [key]: value } : f));

  const toggleFile = (path: string) =>
    setFiles((prev) => {
      const next = new Set(prev);
      next.has(path) ? next.delete(path) : next.add(path);
      setFolder(null);
      return next;
    });

  const selectFolder = (path: string) => {
    setFolder(path);
    setFiles(new Set());
  };

  const onUploaded = (paths: string[]) =>
    setFiles((prev) => {
      const next = new Set(prev);
      paths.forEach((p) => next.add(p));
      setFolder(null);
      return next;
    });

  const canRun = useMemo(
    () => folder !== null || files.size > 0,
    [folder, files]
  );

  const run = async () => {
    if (!form || !canRun) return;
    setSubmitting(true);
    setErr(null);
    try {
      const mode = folder !== null ? "directory" : "files";
      const targets = folder !== null ? [folder] : Array.from(files);
      const { job_id } = await api.createJob({ mode, targets, ...form });
      setJobId(job_id);
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setSubmitting(false);
    }
  };

  if (!options || !form) {
    return <div className="p-8 text-sm opacity-60">…</div>;
  }

  return (
    <div className="mx-auto max-w-6xl p-4">
      <header className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">{t("title")}</h1>
          <p className="text-sm opacity-60">{t("subtitle")}</p>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={locale}
            onChange={(e) => setLocale(e.target.value as Locale)}
            className="rounded border border-slate-300 bg-white px-2 py-1 text-sm dark:border-slate-600 dark:bg-slate-800"
          >
            {LOCALES.map((l) => (
              <option key={l} value={l}>{l}</option>
            ))}
          </select>
          <button
            onClick={toggle}
            className="rounded border border-slate-300 px-2 py-1 text-sm dark:border-slate-600"
            title="theme"
          >
            {dark ? "🌙" : "☀️"}
          </button>
        </div>
      </header>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Left: source selection + options */}
        <section className="space-y-5">
          <div className="rounded-lg border border-slate-200 p-4 dark:border-slate-700">
            <h2 className="mb-3 font-semibold">{t("source")}</h2>
            <div className="mb-3 flex gap-2">
              <Tab active={tab === "browse"} onClick={() => setTab("browse")}>
                {t("browse")}
              </Tab>
              <Tab active={tab === "upload"} onClick={() => setTab("upload")}>
                {t("upload")}
              </Tab>
            </div>
            {tab === "browse" ? (
              <FileBrowser
                selected={files}
                onToggleFile={toggleFile}
                onSelectFolder={selectFolder}
              />
            ) : (
              <Uploader onUploaded={onUploaded} />
            )}

            <p className="mt-3 text-xs opacity-70">
              {folder !== null
                ? `📁 /${folder}`
                : files.size > 0
                ? `${files.size} ${t("selectedFiles")}`
                : t("nothingSelected")}
            </p>
          </div>

          <div className="rounded-lg border border-slate-200 p-4 dark:border-slate-700">
            <h2 className="mb-3 font-semibold">{t("options")}</h2>
            <OptionsForm options={options} form={form} update={update} />
          </div>

          <button
            onClick={run}
            disabled={!canRun || submitting}
            className="w-full rounded-lg bg-indigo-600 px-4 py-2.5 font-medium text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {submitting ? t("processing") : t("process")}
          </button>
          {err && <p className="text-sm text-red-500">{err}</p>}
        </section>

        {/* Right: progress */}
        <section className="rounded-lg border border-slate-200 p-4 dark:border-slate-700">
          {jobId ? (
            <ProgressPanel jobId={jobId} />
          ) : (
            <p className="text-sm opacity-50">{t("waiting")}</p>
          )}
        </section>
      </div>
    </div>
  );
}

function Tab({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={
        "rounded px-3 py-1 text-sm " +
        (active
          ? "bg-indigo-600 text-white"
          : "bg-slate-100 dark:bg-slate-800")
      }
    >
      {children}
    </button>
  );
}
