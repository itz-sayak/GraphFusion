"use client";

import ChatPanel from "@/components/ChatPanel";
import DatasetSidebar from "@/components/DatasetSidebar";
import Header from "@/components/Header";
import Workspace from "@/components/Workspace";
import { StoreProvider } from "@/lib/store";

export default function Home() {
  return (
    <StoreProvider>
      <div className="flex h-screen flex-col">
        <Header />
        <div className="flex min-h-0 flex-1">
          <DatasetSidebar />
          <main className="flex min-w-0 flex-1 flex-col">
            <ChatPanel />
            <Workspace />
          </main>
        </div>
      </div>
    </StoreProvider>
  );
}
