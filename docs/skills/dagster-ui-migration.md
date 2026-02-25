---
name: dagster-ui-migration
description: Blueprint to Dagster UI component migration patterns and TypeScript fixes. Use when migrating Blueprint components or debugging Dagster UI TypeScript errors.
version: 1.0
---

# Dagster UI Migration - Quick Reference

**Canonical Documentation**: See `.github/skills/dagster-ui-migration/SKILL.md` for full reference.

## 5 Most Common Patterns

### 1. Remove Blueprint Props from Styled Components

```tsx
// ❌ WRONG
const Card = styled.div`...`;
<Card elevation={1} striped interactive>

// ✅ CORRECT
<Card>  // Just remove Blueprint props
```

### 2. Wrap Tag for onClick Support

```tsx
// ❌ WRONG
<Tag onClick={handleClick}>Label</Tag>

// ✅ CORRECT
const TagButton = styled.button`
  border: none; background: none; cursor: pointer; padding: 0;
`;

<TagButton onClick={handleClick}>
  <Tag interactive>Label</Tag>
</TagButton>
```

### 3. Fix Intent Type Mismatches

```tsx
// ❌ WRONG
const intent: Intent = "error";

// ✅ CORRECT
const intent: Intent = "danger";  // error → danger
```

### 4. Popover Type Assertion

```tsx
// ❌ TypeScript error despite valid API
<Popover content={menu} isOpen={isOpen}>

// ✅ CORRECT - Workaround
<Popover {...({content: menu, isOpen} as any)}>
```

### 5. Native HTML Replacements

```tsx
// NumericInput → native input
<input type="number" value={x} onChange={e => setX(+e.target.value)} />

// Slider → native range
<input type="range" min={0} max={100} value={x} onChange={e => setX(+e.target.value)} />

// ButtonGroup → flex div
<div style={{display: 'flex', flexDirection: 'column'}}>
  <Button>One</Button>
  <Button>Two</Button>
</div>
```

## Component Mapping

| Blueprint | Dagster/Native | Status |
|-----------|---------------|---------|
| InputGroup | TextInput | Direct replacement |
| Button | Button | Direct replacement |
| Icon | Icon | Direct replacement (no size={16}) |
| Tag | Tag | No onClick - wrap in button |
| Checkbox | Checkbox | Direct replacement |
| NumericInput | `<input type="number">` | Native HTML |
| Slider | `<input type="range">` | Native HTML or Dagster Slider |
| Callout | styled.div | Style manually |
| HTMLTable | styled.table | Native HTML |
| OverlayToaster | console.log | No equivalent |

## Verification

```bash
# Check error count
cd dagster-ui/packages/dsa110
yarn build 2>&1 | grep "error TS" | grep -v "TS5083" | wc -l

# Target: 0 errors
```

---

**Full Documentation**: `.github/skills/dagster-ui-migration/SKILL.md`
**Workflow**: `.agent/workflows/dagster-ui-migration.yaml`
