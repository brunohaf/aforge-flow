import { type JobForm, type Options } from "../api";
import { useI18n } from "../i18n";

interface Props {
  options: Options;
  form: JobForm;
  update: <K extends keyof JobForm>(key: K, value: JobForm[K]) => void;
}

export default function OptionsForm({ options, form, update }: Props) {
  const { t } = useI18n();

  const labelCls = "block text-sm font-medium mb-1";
  const selectCls =
    "w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800";

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-4">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={form.remove_background}
            onChange={(e) => update("remove_background", e.target.checked)}
          />
          {t("removeBackground")}
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={form.optimize_media}
            onChange={(e) => update("optimize_media", e.target.checked)}
          />
          {t("optimizeMedia")}
        </label>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className={labelCls}>{t("category")}</label>
          <select
            className={selectCls}
            value={form.category}
            onChange={(e) => update("category", e.target.value)}
          >
            {options.categories.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>

        <div>
          <label className={labelCls}>{t("rembgModel")}</label>
          <select
            className={selectCls}
            value={form.rembg_model ?? ""}
            onChange={(e) => update("rembg_model", e.target.value || null)}
          >
            <option value="">{t("auto")}</option>
            {options.rembg_models.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>

        <div>
          <label className={labelCls}>{t("imageFormat")}</label>
          <select
            className={selectCls}
            value={form.image_format}
            onChange={(e) => update("image_format", e.target.value)}
          >
            {options.image_formats.map((f) => (
              <option key={f} value={f}>{f}</option>
            ))}
          </select>
        </div>

        <div>
          <label className={labelCls}>{t("videoFormat")}</label>
          <select
            className={selectCls}
            value={form.video_format}
            onChange={(e) => update("video_format", e.target.value)}
          >
            {options.video_formats.map((f) => (
              <option key={f} value={f}>{f}</option>
            ))}
          </select>
        </div>

        <div>
          <label className={labelCls}>{t("compression")}</label>
          <select
            className={selectCls}
            value={form.compression}
            onChange={(e) => update("compression", e.target.value)}
          >
            {options.compression_levels.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>

        <div>
          <label className={labelCls}>{t("accel")}</label>
          <select
            className={selectCls}
            value={form.use_gpu ? "gpu" : "cpu"}
            onChange={(e) => update("use_gpu", e.target.value === "gpu")}
          >
            <option value="gpu">{t("gpu")}</option>
            <option value="cpu">{t("cpu")}</option>
          </select>
        </div>

        <div>
          <label className={labelCls}>{t("workers")}</label>
          <input
            type="number"
            min={1}
            className={selectCls}
            value={form.workers}
            onChange={(e) => update("workers", Math.max(1, Number(e.target.value)))}
          />
        </div>

        <div>
          <label className={labelCls}>{t("outputDir")}</label>
          <input
            type="text"
            className={selectCls}
            value={form.output_dir}
            onChange={(e) => update("output_dir", e.target.value)}
          />
        </div>
      </div>

      <label className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={form.skip_existing}
          onChange={(e) => update("skip_existing", e.target.checked)}
        />
        {t("skipExisting")}
      </label>
    </div>
  );
}
