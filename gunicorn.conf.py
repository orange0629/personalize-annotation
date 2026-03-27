# gunicorn.conf.py
def post_fork(server, worker):
    """Re-open all inherited file handles after fork to avoid shared-fd races."""
    import app as application

    # 重新打开 extracted convs handle（最关键）
    if application._convs_exfh is not None:
        try:
            application._convs_exfh.close()
        except Exception:
            pass
    if application.EXTRACTED_CONVS.exists():
        application._convs_exfh = open(application.EXTRACTED_CONVS, "rb")

    # 重置懒加载的 fallback handles（让它们在 worker 里重新打开）
    application._attrs_fh = None
    application._checklist_fh = None
    application._convs_fh = None