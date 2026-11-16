import { type ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ErrorBoundary } from '@/components/error-boundary';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import NotFound from '@/pages/not-found';
import { Route, Switch, useLocation, Router as WouterRouter } from 'wouter';
import { ExperimentProvider } from '@/lib/experiment-context';
import { Shell } from '@/components/shell';
import LivePage from '@/pages/live';
import SensorsPage from '@/pages/sensors';
import AnalysisPage from '@/pages/analysis';
import GoiPage from '@/pages/goi';
import BenchmarksPage from '@/pages/benchmarks';
import MethodologyPage from '@/pages/methodology';

const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 } },
});

function Router() {
  return (
    <Shell>
      <RoutedErrorBoundary>
        <Switch>
          <Route path="/" component={LivePage} />
          <Route path="/sensors" component={SensorsPage} />
          <Route path="/analysis" component={AnalysisPage} />
          <Route path="/goi" component={GoiPage} />
          <Route path="/benchmarks" component={BenchmarksPage} />
          <Route path="/methodology" component={MethodologyPage} />
          <Route component={NotFound} />
        </Switch>
      </RoutedErrorBoundary>
    </Shell>
  );
}

function RoutedErrorBoundary({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  return <ErrorBoundary resetKey={location}>{children}</ErrorBoundary>;
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}>
          <ExperimentProvider>
            <Router />
          </ExperimentProvider>
        </WouterRouter>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
