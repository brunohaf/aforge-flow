import { useI18n } from "../i18n";
import { LOCALES, type Locale } from "../i18n/locales";

/** Locale picker (pt-BR / en-US). */
export default function LanguageSelect() {
  const { locale, setLocale } = useI18n();
  return (
    <select
      value={locale}
      onChange={(e) => setLocale(e.target.value as Locale)}
      className="rounded border border-slate-300 bg-white px-2 py-1 text-sm dark:border-slate-600 dark:bg-slate-800"
    >
      {LOCALES.map((l) => (
        <option key={l} value={l}>
          {l}
        </option>
      ))}
    </select>
  );
}
