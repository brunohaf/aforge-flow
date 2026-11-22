import { useI18n } from "../i18n";

export default function WarmingUp() {
  const { t } = useI18n();
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-6 p-8">
      <div className="h-14 w-14 animate-spin rounded-full border-4 border-slate-200 border-t-indigo-600 dark:border-slate-700 dark:border-t-indigo-400" />
      <div className="text-center">
        <p className="text-lg font-semibold">{t("warmingUp")}</p>
        <p className="mt-1 animate-pulse text-sm opacity-60">{t("warmingUpHint")}</p>
      </div>
    </div>
  );
}
