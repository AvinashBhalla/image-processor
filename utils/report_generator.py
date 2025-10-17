class DuplicateReportGenerator:
    """
    Generate reports for duplicate detection results
    """
    
    def __init__(self):
        self.report_data = {}
    
    def generate_report(self, 
                       duplicates: Dict[str, List[str]], 
                       output_path: str = "duplicate_report.html"):
        """
        Generate HTML report with duplicate groups
        """
        html_content = self._create_html_template()
        
        total_duplicates = sum(len(dups) for dups in duplicates.values())
        space_savings = self._calculate_space_savings(duplicates)
        
        # Add statistics
        stats_html = f"""
        <div class="statistics">
            <h2>Duplicate Detection Summary</h2>
            <p><strong>Total duplicate groups:</strong> {len(duplicates)}</p>
            <p><strong>Total duplicate files:</strong> {total_duplicates}</p>
            <p><strong>Potential space savings:</strong> {space_savings / (1024**2):.2f} MB</p>
        </div>
        """
        
        # Add duplicate groups
        groups_html = "<div class='duplicate-groups'>"
        
        for idx, (representative, duplicates_list) in enumerate(duplicates.items()):
            groups_html += self._create_group_html(idx, representative, duplicates_list)
        
        groups_html += "</div>"
        
        # Combine and save
        final_html = html_content.replace("{{STATS}}", stats_html)
        final_html = final_html.replace("{{GROUPS}}", groups_html)
        
        with open(output_path, 'w') as f:
            f.write(final_html)
        
        print(f"Report generated: {output_path}")
    
    def _calculate_space_savings(self, duplicates: Dict[str, List[str]]) -> int:
        """Calculate potential space savings from removing duplicates"""
        total_size = 0
        
        for representative, dups in duplicates.items():
            for dup_path in dups:
                try:
                    total_size += Path(dup_path).stat().st_size
                except:
                    continue
        
        return total_size
    
    def _create_group_html(self, idx: int, 
                          representative: str, 
                          duplicates: List[str]) -> str:
        """Create HTML for a duplicate group"""
        group_html = f"""
        <div class="duplicate-group">
            <h3>Group {idx + 1}</h3>
            <div class="representative">
                <h4>Keep (Representative)</h4>
                <img src="file://{representative}" />
                <p>{Path(representative).name}</p>
                <p class="file-info">Size: {Path(representative).stat().st_size / 1024:.2f} KB</p>
            </div>
            <div class="duplicates-list">
                <h4>Duplicates ({len(duplicates)}) - Consider Deleting</h4>
        """
        
        for dup_path in duplicates:
            try:
                size = Path(dup_path).stat().st_size / 1024
                group_html += f"""
                <div class="duplicate-item">
                    <img src="file://{dup_path}" />
                    <p>{Path(dup_path).name}</p>
                    <p class="file-info">Size: {size:.2f} KB</p>
                    <button onclick="deleteFile('{dup_path}')">Delete</button>
                </div>
                """
            except:
                continue
        
        group_html += """
            </div>
        </div>
        """
        
        return group_html
    
    def _create_html_template(self) -> str:
        """HTML template for report"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Duplicate Detection Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .statistics { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                .duplicate-group { border: 1px solid #ccc; margin: 20px 0; padding: 15px; }
                .representative { background: #e8f5e9; padding: 10px; }
                .duplicates-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; margin-top: 10px; }
                .duplicate-item { border: 1px solid #ddd; padding: 10px; text-align: center; }
                img { max-width: 100%; height: auto; max-height: 200px; object-fit: contain; }
                button { background: #f44336; color: white; border: none; padding: 5px 10px; cursor: pointer; }
                button:hover { background: #d32f2f; }
                .file-info { font-size: 0.9em; color: #666; }
            </style>
        </head>
        <body>
            <h1>Image Duplicate Detection Report</h1>
            {{STATS}}
            {{GROUPS}}
        </body>
        </html>
        """