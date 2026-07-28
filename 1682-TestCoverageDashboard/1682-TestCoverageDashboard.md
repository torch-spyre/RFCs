# Test Coverage Dashboard Implementation

**Authors:**
* @thanmaiboddoju (Thanmai Boddoju, IBM Infrastructure, India)
* @anubhavjana (Anubhav Jana, IBM Research, India)

## **Summary**
Request approval to adopt an enhanced test coverage dashboard that provides comprehensive visualization and analysis capabilities for Spyre pytest test results through an interactive, self-contained browser-based interface.

## **Motivation**
The current test reporting system lacks interactive visualization capabilities. Manual analysis of XML reports is time-consuming with limited filtering and search. This dashboard will:

* **Improve Developer Productivity**: Reduce time spent analyzing test results and aid in debugging
* **Enable Better Decision Making**: Provide clear visibility into test coverage and failure patterns
* **Streamline CI/CD Integration**: Zero-infrastructure solution integrating seamlessly with existing pipelines
* **Track Model Enablement**: The dashboard will provide op/module test success ratio on a per-model basis, enabling teams to understand which operations and modules are working correctly for each model and identify areas requiring attention

## **Proposed Implementation**

### Architecture
A self-contained, single-file HTML dashboard that runs entirely in the browser with no backend dependencies.

**Note on Implementation Approach**: As this dashboard is designed for internal usage within the development team, it has been developed as a single HTML file for simplicity, ease of deployment, and minimal infrastructure requirements. In the future, if this tool is extended for client-facing usage or requires more advanced features, it can be migrated to a framework-based application (e.g., React) to support better scalability, maintainability, and enhanced functionality.

### Features
**Navigation Tabs:**
- **Test Overview**: Metrics, status distribution chart, top failures, pass rate by model
- **Test Details**: Comprehensive listing with filters, search, pagination, expandable errors
- **Coverage Explorer**: Dynamic grouping by Model Tag, Test Method, Op Name, Dtype, or Test Class
- **Ops and Module Coverage Metrics**: Track operation and module test success ratios

**Key Capabilities:**
- Real-time filtering and search across test names
- Export to CSV and PDF
- Status-based filtering (passed, failed, xfail, xpass, skipped)
- Per-model basis tracking for granular enablement insights

**Technical Stack:**
- HTML5 + CSS3 + Vanilla JavaScript
- Chart.js for visualizations
- Single self-contained file
- No server/backend or build process required
- Direct integration with GitHub Actions



**Current Status**: Awaiting team review and approval for official adoption

### Project Timeline

### Phase 1: Initial Development (Weeks 1-2) [In progress]
- Gather requirements and finalize design
- Develop core dashboard functionality
- Implement basic visualization features
- Create prototype for review
- Initial testing with sample data

### Phase 2: Enhancement & Validation (Week 3)
- Incorporate feedback from Phase 1
- Add additional features as identified
- Integrate with the CI/CD pipeline
- QA team validation


