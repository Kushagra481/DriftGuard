import { pgTable, text, integer, doublePrecision, timestamp, jsonb } from "drizzle-orm/pg-core";

// Numerical arrays and diagnostics are structured experiment data, not files.
export const experimentsTable = pgTable("driftguard_experiments", {
  id: text("id").primaryKey(),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  preset: text("preset").notNull(),
  seed: integer("seed").notNull(),
  duration: doublePrecision("duration").notNull(),
  ate: doublePrecision("ate").notNull(),
  result: jsonb("result").notNull(),
  details: jsonb("details").notNull(),
});
