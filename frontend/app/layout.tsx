import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Zettlecast',
  description: 'Digital Zettelkasten Middleware - Local-First AI Knowledge System',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
