import { NavLink, Outlet } from "react-router-dom";
import styles from "./AppShell.module.css";

// Nav scaffolding for the phases after this one — track record, CLV report,
// status, config each get a route here later, sharing this same shell/theme/
// data-client layer. Only "Matchup" is wired to a real page today.
const NAV_ITEMS = [
  { to: "/", label: "Matchup" },
  { to: "/track-record", label: "Track Record" },
  { to: "/clv", label: "CLV Report" },
  { to: "/status", label: "Status" },
];

export function AppShell() {
  return (
    <div className={styles.shell}>
      <aside className={styles.sidebar}>
        <div className={styles.brand}>FBQ</div>
        <nav className={styles.nav}>
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => (isActive ? `${styles.link} ${styles.active}` : styles.link)}
              end={item.to === "/"}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className={styles.main}>
        <Outlet />
      </main>
    </div>
  );
}
