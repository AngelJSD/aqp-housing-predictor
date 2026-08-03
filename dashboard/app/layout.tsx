import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Arequipa Housing — Observability",
  description: "MLOps observability dashboard for the Arequipa housing price predictor.",
};

const NAV_LINKS = [
  { href: "/", label: "Overview" },
  { href: "/predictor", label: "Predictor" },
  { href: "/models", label: "Models" },
  { href: "/monitoring", label: "Monitoring" },
  { href: "/drift", label: "Drift" },
  { href: "/features", label: "Feature catalog" },
];

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-zinc-50 dark:bg-black">
        <nav className="flex gap-6 border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
          {NAV_LINKS.map((link) => (
            <Link key={link.href} href={link.href} className="text-sm font-medium text-zinc-700 hover:text-zinc-950 dark:text-zinc-300 dark:hover:text-zinc-50">
              {link.label}
            </Link>
          ))}
        </nav>
        <main className="flex-1 p-6">{children}</main>
      </body>
    </html>
  );
}
