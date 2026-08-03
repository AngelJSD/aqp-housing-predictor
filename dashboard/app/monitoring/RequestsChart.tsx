"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

type Row = { label: string; requests: number; avgLatencyMs: number };

export function RequestsChart({ rows }: { rows: Row[] }) {
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="label" tick={{ fontSize: 11 }} />
          <YAxis />
          <Tooltip />
          <Bar dataKey="requests" fill="#6366f1" name="Requests" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
