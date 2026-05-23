import type { Metadata } from "next";
import { IBM_Plex_Mono, Inter } from "next/font/google";
import "./globals.css";

const sans = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
});

const mono = IBM_Plex_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "Job Engine",
  description: "Find opportunities ranked by $/hour.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable} h-full`} suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var t=localStorage.getItem("job-engine-theme");if(t==="dark")document.documentElement.classList.add("dark");}catch(e){}})();`,
          }}
        />
      </head>
      <body className="min-h-full">{children}</body>
    </html>
  );
}
