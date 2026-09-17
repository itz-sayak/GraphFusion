import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GraphFusion",
  description: "Conversational multi-format dataset integration with graph-based matching, entity resolution and provenance.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="font-sans antialiased">{children}</body>
    </html>
  );
}
