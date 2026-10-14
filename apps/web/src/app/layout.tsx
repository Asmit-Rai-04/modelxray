import type { Metadata } from "next";
import "./globals.css";
import "./instrument.css";

export const metadata: Metadata = {
  title: "ModelXray — ML Model Assurance",
  description:
    "Systematic ML model failure discovery, validation, and regression analysis.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `try{var t=localStorage.getItem('modelxray-theme');document.documentElement.dataset.theme=t==='light'||t==='dark'?t:matchMedia('(prefers-color-scheme: light)').matches?'light':'dark'}catch{}`,
          }}
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
