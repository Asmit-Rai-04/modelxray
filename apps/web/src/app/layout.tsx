import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ModelXray — ML Model Assurance",
  description: "Systematic ML model failure discovery, validation, and regression analysis.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
