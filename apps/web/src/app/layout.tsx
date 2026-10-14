import type { Metadata } from "next";
import localFont from "next/font/local";
import "./tokens.css";
import "./globals.css";

const geist = localFont({ src: "../../node_modules/@fontsource-variable/geist/files/geist-latin-wght-normal.woff2", variable: "--font-geist", display: "swap", weight: "100 900" });
const plex = localFont({ src: "../../node_modules/@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-400-normal.woff2", variable: "--font-plex", display: "swap", weight: "400", preload: false });

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
    <html lang="en" className={`${geist.variable} ${plex.variable}`} suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `try{var t=localStorage.getItem('modelxray-theme');document.documentElement.dataset.theme=t==='light'||t==='dark'?t:matchMedia('(prefers-color-scheme: light)').matches?'light':'dark'}catch{document.documentElement.dataset.theme=matchMedia('(prefers-color-scheme: light)').matches?'light':'dark'}`,
          }}
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
