import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { LOCALES, STRINGS, type Locale } from "./locales";

interface I18n {
  locale: Locale;
  setLocale: (l: Locale) => void;
  t: (key: string) => string;
}

const I18nContext = createContext<I18n | null>(null);

function initialLocale(): Locale {
  const saved = localStorage.getItem("aforge-locale") as Locale | null;
  if (saved && LOCALES.includes(saved)) return saved;
  return navigator.language.startsWith("pt") ? "pt-BR" : "en-US";
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(initialLocale);

  const setLocale = useCallback((l: Locale) => {
    localStorage.setItem("aforge-locale", l);
    setLocaleState(l);
  }, []);

  const t = useCallback(
    (key: string) => STRINGS[locale][key] ?? key,
    [locale]
  );

  const value = useMemo(() => ({ locale, setLocale, t }), [locale, setLocale, t]);
  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18n {
  const ctx = useContext(I18nContext);
  if (!ctx) throw new Error("useI18n must be used within I18nProvider");
  return ctx;
}
