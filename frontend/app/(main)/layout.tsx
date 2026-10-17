import Sidebar from '@/components/Sidebar';

// Pages with the sidebar; the graph has its own full-screen view (app/graph)
export default function MainLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main-content">{children}</main>
    </div>
  );
}
